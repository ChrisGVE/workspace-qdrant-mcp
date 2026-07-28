//! N43's transport half, bin-local (ARCH rev15 §9.1: N43 lives in the bin's own
//! source tree, not in a crate).
//!
//! Newline-delimited JSON-RPC 2.0 over stdio: one request per line in, one
//! response per line out. `P04-GT001-WO011` implements the three methods the
//! walking skeleton needs -- `initialize`, `tools/list`, `tools/call` -- for the
//! one tool it serves. The remaining seven tools and the resource channel arrive
//! with N43's slice (`P04-GT061`).
//!
//! An MCP SDK is deliberately not taken on yet: the skeleton needs three methods,
//! and the choice of SDK is a decision N43's slice should make with the whole
//! surface in view rather than one inherited from the first tool. Recorded as
//! debt in `SCAFFOLD.md` §7.

use std::io::{BufRead, Write};
use std::path::Path;

use serde_json::{json, Value};
use wqm_client::Client;
use wqm_common::envelope::{Envelope, Notice, NoticeCode, Severity};
use wqm_common::names::{negotiate, Negotiated, SUPPORTED_PROTOCOLS};

use crate::query::{self, Refusal};
use crate::status;

/// What one session needs in order to answer: the daemon seam `status` crosses,
/// and the store `query` reads. Both are addresses this build refuses to default
/// (CR-007's first defect), so both are optional and their absence is *reported*
/// rather than guessed at.
pub struct Surfaces<'a> {
    /// The N49 client seam.
    pub client: &'a Client,
    /// The async runtime the seam's calls block on.
    pub runtime: &'a tokio::runtime::Runtime,
    /// The SQLite store the read leg queries, when one was passed.
    pub store: Option<&'a Path>,
}

/// JSON-RPC's "method not found".
const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC's "invalid params" -- the carrier for `invalid_argument` (§4.1).
const INVALID_PARAMS: i64 = -32602;

/// The session's state: what was negotiated, and whether the client still owes a
/// downgrade notice.
struct Session {
    handshake: Negotiated,
    downgrade_undisclosed: bool,
}

impl Session {
    fn new() -> Self {
        Session {
            handshake: negotiate(None),
            downgrade_undisclosed: false,
        }
    }
}

/// Serve one stdio session until the input closes.
pub fn serve(
    input: impl BufRead,
    mut output: impl Write,
    surfaces: &Surfaces<'_>,
) -> std::io::Result<()> {
    let mut session = Session::new();

    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let Some(response) = handle_line(&line, &mut session, surfaces) else {
            // A notification (no `id`) gets no response, per JSON-RPC.
            continue;
        };
        writeln!(output, "{response}")?;
        output.flush()?;
    }
    Ok(())
}

/// Parse and dispatch one line. `None` means "no response is owed".
fn handle_line(line: &str, session: &mut Session, surfaces: &Surfaces<'_>) -> Option<Value> {
    let request: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        // A line that is not JSON has no id to answer against; JSON-RPC's parse
        // error uses a null id.
        Err(e) => {
            return Some(error_response(
                Value::Null,
                -32700,
                &format!("the request was not valid JSON: {e}"),
                Value::Null,
            ))
        }
    };

    let id = request.get("id").cloned();
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let params = request.get("params").cloned().unwrap_or(Value::Null);

    // No id = a notification. Handled for effect, answered with nothing.
    let id = id?;

    Some(match method {
        "initialize" => initialize(id, &params, session),
        "tools/list" => result_response(id, tools_list()),
        "tools/call" => tools_call(id, &params, session, surfaces),
        other => error_response(
            id,
            METHOD_NOT_FOUND,
            &format!("this server does not implement `{other}`"),
            json!({"implemented": ["initialize", "tools/list", "tools/call"]}),
        ),
    })
}

/// `initialize` -- negotiate, never echo (§5.1).
fn initialize(id: Value, params: &Value, session: &mut Session) -> Value {
    let requested = params.get("protocolVersion").and_then(Value::as_str);
    session.handshake = negotiate(requested);
    session.downgrade_undisclosed = session.handshake.downgraded;

    result_response(
        id,
        json!({
            "protocolVersion": session.handshake.served,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": env!("CARGO_BIN_NAME"),
                "version": env!("CARGO_PKG_VERSION"),
            },
        }),
    )
}

/// The tools this build serves, spelled once. `tools/list` advertises them and
/// `capabilities.tools` reports them, so the two renderings cannot drift.
pub const SERVED_TOOLS: [&str; 2] = ["status", "query"];

/// `tools/list` -- exactly what this build serves, which is two of eight.
///
/// The sealed inventory has eight (§1.2). Advertising eight and serving two would
/// be the "declared feature that silently does nothing" class the surface design
/// exists to end, so the list is short and true. The same rule applies *inside*
/// each schema: `query`'s sealed schema carries `plan`, `cursor` and `shape`, and
/// this build declares none of them, because a parameter that is accepted and
/// ignored is the same failure one level down.
fn tools_list() -> Value {
    json!({"tools": [status_tool(), query_tool()]})
}

fn status_tool() -> Value {
    json!({
        "name": "status",
        "description": "Report what this server can currently do and what is degraded: \
                        daemon reachability, index freshness, the embedding provider, the \
                        four collections, the project your working directory resolves to, \
                        and the exact set of query clauses this build executes.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "probe": {
                    "type": "boolean",
                    "description": "Actively probe the embedding provider and the daemon \
                                    rather than reporting cached state (default: false)."
                }
            }
        }
    })
}

/// `query`'s description is written against what THIS build executes.
///
/// §5.4's rule is that every example in a tool description draws only from the
/// guaranteed core, "because a fixed description cannot outrun a variable build if
/// its examples only use what is invariant". The sealed description's examples
/// (`SELECT chunk WHERE path~'src/**'`, `FROM library['tokio']`) are outside this
/// build's core, so serving it verbatim would ship an example that does not run --
/// which is the exact defect (AS-F019) that the rule exists to prevent.
/// The description, with its one worked example built from N8's registry rather
/// than spelled here.
///
/// §5.4 requires every shipped example to be executable on the build that ships
/// it, and L-8 makes an example that does not run a build failure. An example is
/// therefore not prose: it is an assertion about the parser, and it has to be
/// built from the same registry the parser matches against. Two spellings of one
/// collection name is precisely how a shipped example stops running without
/// anyone noticing.
fn query_description() -> String {
    format!(
        "Find matches in the indexed corpus using one query language. Returns ranked \
         results, the plan that ran, and every default that was applied to your request. \
         This build executes a narrow subset of the language: \
         `SELECT TEXT <object> FROM <collection> WHERE q MATCH '<text>' [LIMIT n]`, where \
         the match is a literal substring of at least three characters. Example, \
         executable on this build: `SELECT TEXT note FROM {} WHERE q MATCH 'skeleton'`. \
         Call `status` for the clause set this build executes; a clause outside it is \
         refused by name rather than silently ignored.",
        wqm_common::names::Collection::Scratchpad.name()
    )
}

fn query_tool() -> Value {
    json!({
        "name": "query",
        "description": query_description(),
        "inputSchema": {
            "type": "object",
            "additionalProperties": false,
            "required": ["q"],
            "properties": {
                "q": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The query, in wqm's SQL-family language; the executable shape \
                                    is in this tool's description. Every default applied to it is \
                                    named back to you in the response's defaults_applied."
                },
                "limit": {
                    "type": "integer", "minimum": 1, "maximum": 200,
                    "description": "Maximum results to return (default: 10). Supplying this AND a \
                                    LIMIT clause in q is an error, not a silent override."
                }
            }
        }
    })
}

/// `tools/call` -- dispatch, then the sealed envelope either way.
fn tools_call(id: Value, params: &Value, session: &mut Session, surfaces: &Surfaces<'_>) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
    let started = std::time::Instant::now();

    let envelope = match name {
        "status" => match status_envelope(&arguments, session, surfaces, started) {
            Ok(envelope) => envelope,
            Err(response) => return response(id),
        },
        "query" => match query_envelope(&arguments, surfaces, started) {
            Ok(envelope) => envelope,
            Err(response) => return response(id),
        },
        other => {
            return error_response(
                id,
                INVALID_PARAMS,
                &format!("`{other}` is not a tool this build serves"),
                json!({
                    "code": "invalid_argument",
                    "details": {"parameter": "name", "got": other, "expected": SERVED_TOOLS},
                    "retryable": false,
                }),
            )
        }
    };

    tool_result(id, &disclose_downgrade(envelope, session))
}

/// A protocol-level refusal, deferred until the request id is in hand.
type ProtocolRefusal = Box<dyn FnOnce(Value) -> Value>;

fn status_envelope(
    arguments: &Value,
    session: &Session,
    surfaces: &Surfaces<'_>,
    started: std::time::Instant,
) -> Result<Envelope, ProtocolRefusal> {
    let probe = match arguments.get("probe") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(other) => {
            let got = other.clone();
            return Err(Box::new(move |id| {
                error_response(
                    id,
                    INVALID_PARAMS,
                    "`probe` must be a boolean",
                    json!({
                        "code": "invalid_argument",
                        "details": {"parameter": "probe", "got": got, "expected": "boolean"},
                        "retryable": false,
                    }),
                )
            }));
        }
    };

    let report = surfaces.runtime.block_on(surfaces.client.status(probe));
    let elapsed_ms = started.elapsed().as_millis() as u64;

    Ok(match report {
        Ok(report) => Envelope::success(
            "status",
            elapsed_ms,
            status::build(&report, &session.handshake),
        ),
        Err(e) => Envelope::failure(
            "status",
            elapsed_ms,
            wqm_common::envelope::ToolError {
                // Not `backend_unavailable`: reaching this arm means the call
                // itself faulted, not that the daemon is down -- an absent daemon
                // is data and never lands here (§4.3, and wqm-client's contract).
                code: wqm_common::envelope::ErrorCode::Internal,
                message: e.to_string(),
                details: json!({"correlation_id": Value::Null}),
                retryable: false,
            },
        ),
    })
}

fn query_envelope(
    arguments: &Value,
    surfaces: &Surfaces<'_>,
    started: std::time::Instant,
) -> Result<Envelope, ProtocolRefusal> {
    let answer = query::call(surfaces.store, arguments);
    let elapsed_ms = started.elapsed().as_millis() as u64;

    match answer {
        Ok(answer) => {
            let mut envelope = Envelope::success("query", elapsed_ms, answer.data);
            for applied in answer.defaults {
                envelope = envelope.with_default(applied);
            }
            Ok(envelope)
        }
        Err(Refusal::Tool(error)) => Ok(Envelope::failure("query", elapsed_ms, *error)),
        // §4.1: `invalid_argument` has no envelope at all -- it rides the JSON-RPC
        // error's `data` member, which is the one level where the two shapes differ.
        Err(Refusal::Protocol { message, details }) => Err(Box::new(move |id| {
            error_response(id, INVALID_PARAMS, &message, details)
        })),
    }
}

/// §4.4's `protocol_downgraded`, fired once, on the FIRST tool call of a
/// downgraded session (§5.1) -- whichever tool that call names.
fn disclose_downgrade(envelope: Envelope, session: &mut Session) -> Envelope {
    if !session.downgrade_undisclosed {
        return envelope;
    }
    session.downgrade_undisclosed = false;
    envelope.with_notice(downgrade_notice(&session.handshake))
}

/// §4.4's `protocol_downgraded`, fired once, on the first tool call of a
/// downgraded session (§5.1).
fn downgrade_notice(handshake: &Negotiated) -> Notice {
    let requested = handshake.requested.clone().unwrap_or_default();
    Notice {
        code: NoticeCode::ProtocolDowngraded,
        severity: Severity::Warn,
        message: format!(
            "this server does not speak `{requested}`; it served `{}` instead",
            handshake.served
        ),
        details: json!({
            "requested": requested,
            "served": handshake.served,
            "supported": SUPPORTED_PROTOCOLS,
        }),
    }
}

/// Wrap an envelope as an MCP `CallToolResult`. The envelope rides
/// `structuredContent`, with the same JSON in `content` as text so a client that
/// reads only the text channel still gets the whole answer.
fn tool_result(id: Value, envelope: &Envelope) -> Value {
    let body = serde_json::to_value(envelope).unwrap_or(Value::Null);
    let text = serde_json::to_string(&body).unwrap_or_default();
    result_response(
        id,
        json!({
            "content": [{"type": "text", "text": text}],
            "structuredContent": body,
            // `isError` mirrors the envelope's own `ok`, read from it rather than
            // re-derived, so the two can never disagree.
            "isError": !envelope.is_ok(),
        }),
    )
}

fn result_response(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn error_response(id: Value, code: i64, message: &str, data: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": data}})
}

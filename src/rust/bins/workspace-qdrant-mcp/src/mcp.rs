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

use serde_json::{json, Value};
use wqm_client::Client;
use wqm_common::envelope::{Envelope, Notice, NoticeCode, Severity};
use wqm_common::names::{negotiate, Negotiated, SUPPORTED_PROTOCOLS};

use crate::status;

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
    client: &Client,
    runtime: &tokio::runtime::Runtime,
) -> std::io::Result<()> {
    let mut session = Session::new();

    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let Some(response) = handle_line(&line, &mut session, client, runtime) else {
            // A notification (no `id`) gets no response, per JSON-RPC.
            continue;
        };
        writeln!(output, "{response}")?;
        output.flush()?;
    }
    Ok(())
}

/// Parse and dispatch one line. `None` means "no response is owed".
fn handle_line(
    line: &str,
    session: &mut Session,
    client: &Client,
    runtime: &tokio::runtime::Runtime,
) -> Option<Value> {
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
        "tools/call" => tools_call(id, &params, session, client, runtime),
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

/// `tools/list` -- exactly what this build serves, which is one tool.
///
/// The sealed inventory has eight (§1.2). Advertising eight and serving one would
/// be the "declared feature that silently does nothing" class the surface design
/// exists to end, so the list is short and true.
fn tools_list() -> Value {
    json!({
        "tools": [{
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
        }]
    })
}

/// `tools/call` -- the one tool, in the sealed envelope.
fn tools_call(
    id: Value,
    params: &Value,
    session: &mut Session,
    client: &Client,
    runtime: &tokio::runtime::Runtime,
) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    if name != "status" {
        return error_response(
            id,
            INVALID_PARAMS,
            &format!("`{name}` is not a tool this build serves"),
            json!({
                "code": "invalid_argument",
                "details": {"parameter": "name", "got": name, "expected": ["status"]},
                "retryable": false,
            }),
        );
    }

    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
    let probe = match arguments.get("probe") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(other) => {
            return error_response(
                id,
                INVALID_PARAMS,
                "`probe` must be a boolean",
                json!({
                    "code": "invalid_argument",
                    "details": {"parameter": "probe", "got": other, "expected": "boolean"},
                    "retryable": false,
                }),
            )
        }
    };

    let started = std::time::Instant::now();
    let report = runtime.block_on(client.status(probe));
    let elapsed_ms = started.elapsed().as_millis() as u64;

    let envelope = match report {
        Ok(report) => {
            let mut envelope = Envelope::success(
                "status",
                elapsed_ms,
                status::build(&report, &session.handshake),
            );
            if session.downgrade_undisclosed {
                session.downgrade_undisclosed = false;
                envelope = envelope.with_notice(downgrade_notice(&session.handshake));
            }
            envelope
        }
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
    };

    tool_result(id, &envelope)
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

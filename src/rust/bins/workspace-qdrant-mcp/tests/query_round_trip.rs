//! `P04-GT001-WO013` -- the read, driven through the real binary.
//!
//! The acceptance is about a PATH: `S2 bin -> planner -> executor -> N41's read
//! leg -> §3.1's envelope`. A test that called those modules in-process would
//! prove the modules and not the path, which is why every case here spawns the
//! shipped executable and talks newline-delimited JSON-RPC to it over a pipe --
//! WO011's precedent, and v0.1's `QdrantTestContainer` as the standing reminder
//! that code with no caller can be wrong for a year (`WO007`).
//!
//! The store is a file this test seeds through the committed fixture, so what the
//! binary reads is what `WO012` writes -- not a second copy of the schema.
//!
//! The load-bearing case is [`the_read_answers_with_no_daemon_running`]: reads are
//! in-process by architectural decision (ARCH rev15 §9.1 decision B), so `query`
//! answering while nothing is listening on the daemon socket is the decision
//! observed rather than asserted.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use wqm_common::names::{Collection, WriteTarget};
use wqm_test_harness::scratchpad_fixture::{at_path, seed, SeedNote};

/// The MCP server binary under test.
const MCP_BIN: &str = env!("CARGO_BIN_EXE_workspace-qdrant-mcp-v2");

/// The opaque branch token. N3 owns the real `BRANCH_NONE_ID`; the store
/// round-trips whatever it is given, so the test supplies a stand-in.
const BRANCH: &str = "branch-none-placeholder";

/// The seven keys §3.1 requires on every tool result, every time.
const ENVELOPE_KEYS: [&str; 7] = [
    "ok",
    "tool",
    "elapsed_ms",
    "defaults_applied",
    "notices",
    "data",
    "error",
];

#[test]
fn tools_list_advertises_two_tools_and_no_parameter_it_ignores() {
    let session = Session::new("list");
    let responses = session.run(&[json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/list"
    })]);

    let tools = responses[0]["result"]["tools"]
        .as_array()
        .expect("tools is an array");
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(names, ["status", "query"]);

    // The sealed `query` schema carries `plan`, `cursor` and `shape` as well.
    // This build declares none of them, because a parameter that is accepted and
    // ignored is the same failure as a tool that is advertised and does nothing.
    let query = tools.iter().find(|t| t["name"] == "query").expect("query");
    let mut properties: Vec<&str> = query["inputSchema"]["properties"]
        .as_object()
        .expect("properties")
        .keys()
        .map(String::as_str)
        .collect();
    properties.sort_unstable();
    assert_eq!(properties, ["limit", "q"]);
    assert_eq!(query["inputSchema"]["additionalProperties"], json!(false));

    // §5.4 / L-8: a shipped example must be executable on the build that ships
    // it. This one is executed below, in `the_read_answers_with_no_daemon_running`.
    let description = query["description"].as_str().expect("a description");
    assert!(
        description.contains(&example_query()),
        "the description's worked example must be the query this build runs: \
         {description}"
    );
}

/// The whole path, with the daemon absent -- which is the point.
#[test]
fn the_read_answers_with_no_daemon_running() {
    let session = Session::new("read").seeded(&notes());
    let envelope = session.query(&example_query());

    for key in ENVELOPE_KEYS {
        assert!(
            envelope.get(key).is_some(),
            "the envelope is missing `{key}`: {envelope}"
        );
    }
    assert_eq!(envelope.as_object().unwrap().len(), 7, "{envelope}");
    assert_eq!(envelope["ok"], json!(true), "{envelope}");
    assert_eq!(envelope["error"], Value::Null);
    assert_eq!(envelope["tool"], json!("query"));

    let data = &envelope["data"];
    let results = data["results"].as_array().expect("results is an array");
    assert_eq!(results.len(), 1, "one note carries the term: {data}");

    let hit = &results[0];
    assert_eq!(hit["object"], json!("note"));
    assert_eq!(hit["rank"], json!(1));
    assert_eq!(
        hit["score"],
        Value::Null,
        "one leg publishes no comparable score; `rank` is the orderable field"
    );
    assert_eq!(hit["text"], json!(notes()[0].content));
    assert_eq!(hit["location"]["branch"], json!(BRANCH));
    assert_eq!(hit["location"]["unit"]["level"], json!("document"));
    assert_eq!(
        hit["location"]["collection"],
        json!(Collection::Scratchpad.deployed_name()),
        "the DEPLOYED name is what was read, and the -v2 knob exists to keep that \
         visible"
    );

    // The page was fully seen, so the total is counted and the digest is real.
    assert_eq!(data["page"]["matched"], json!(1));
    assert_eq!(data["page"]["matched_exact"], json!(true));
    assert_eq!(data["page"]["has_more"], json!(false));
    assert_eq!(data["page"]["cursor"], Value::Null);
    assert!(data["digest"]
        .as_str()
        .is_some_and(|d| d.starts_with("sha256:")));
    assert_eq!(
        data["sources_searched"],
        json!([Collection::Scratchpad.deployed_name()])
    );
    assert_eq!(data["scoring"]["scale"], json!("none"));
}

/// The plan is the response's memory of what ran, and on this build the two
/// absences ARE the design: no fusion step, no dense leg.
#[test]
fn the_returned_plan_reports_one_leg_and_no_fusion() {
    let session = Session::new("plan").seeded(&notes());
    let plan = session.query(&example_query())["data"]["plan"].clone();

    assert_eq!(plan["mode"], json!("text"));
    assert_eq!(plan["object"], json!("note"));
    assert_eq!(plan["limit"], json!(10));
    assert_eq!(plan["strict"], json!(false));
    assert_eq!(plan["legs"].as_array().map(Vec::len), Some(1));
    assert_eq!(plan["legs"][0]["method"], json!("trigram"));
    assert_eq!(
        plan["fuse"],
        Value::Null,
        "N4 takes a vector of legs and needs N17 dense scores, so a single-leg \
         plan does not reach fusion at all: {plan}"
    );
    assert_eq!(plan["expand"], Value::Null);
    assert_eq!(plan["rerank"], Value::Null);
    assert_eq!(plan["filters"], json!([]));
}

/// L-2: the defaults this call applied, named back to the caller.
#[test]
fn the_defaults_the_call_applied_are_disclosed() {
    let session = Session::new("defaults").seeded(&notes());
    let envelope = session.query(&example_query());

    let applied: Vec<&str> = envelope["defaults_applied"]
        .as_array()
        .expect("an array, always present")
        .iter()
        .map(|row| row["parameter"].as_str().unwrap_or_default())
        .collect();
    assert!(applied.contains(&"limit"), "{envelope}");
    assert!(applied.contains(&"shape"), "{envelope}");
}

/// The tokenizer is `trigram`, and the plan says `trigram`. A word-token index
/// would fail exactly here -- on the substring a caller writes *because* the
/// declaration promised substring matching.
#[test]
fn a_substring_match_finds_the_note_the_word_index_would_miss() {
    let session = Session::new("trigram").seeded(&notes());
    let envelope = session.query(&query_for("kelet"));
    assert_eq!(envelope["ok"], json!(true), "{envelope}");
    assert_eq!(
        envelope["data"]["results"].as_array().map(Vec::len),
        Some(1),
        "`kelet` is a substring of `skeleton`, which is what `trigram` means"
    );
}

/// A default-mode query needs an embedder this build does not have. Serving it
/// with the text leg would answer a different question than the one asked.
#[test]
fn a_semantic_query_is_refused_by_name_rather_than_served_as_text() {
    let session = Session::new("semantic").seeded(&notes());
    let envelope = session.query(&format!(
        "SELECT note FROM {} WHERE q MATCH 'skeleton'",
        Collection::Scratchpad.name()
    ));

    assert_eq!(envelope["ok"], json!(false), "{envelope}");
    assert_eq!(envelope["data"], Value::Null);
    assert_eq!(envelope["error"]["code"], json!("grammar_unsupported"));
    assert_eq!(
        envelope["error"]["details"]["capability_key"],
        json!("modes")
    );
    assert_eq!(envelope["error"]["retryable"], json!(false));
}

/// N35's `grep_eligible` is false for this collection, so the regex leg is gated
/// off before availability is consulted -- the axis exercised by being false.
#[test]
fn a_regex_query_is_refused_by_the_profile_axis() {
    let session = Session::new("regex").seeded(&notes());
    let envelope = session.query(&format!(
        "SELECT REGEX note FROM {} WHERE q MATCH 'ske.*ton'",
        Collection::Scratchpad.name()
    ));

    assert_eq!(envelope["ok"], json!(false), "{envelope}");
    assert_eq!(
        envelope["error"]["details"]["capability_key"],
        json!("grep_eligible")
    );
}

/// The store is an address this build refuses to default (CR-007's first defect),
/// so its absence is REPORTED. The server still starts and still answers.
#[test]
fn a_query_without_a_store_reports_the_missing_backend() {
    let session = Session::new("nostore");
    let envelope = session.query(&example_query());

    assert_eq!(envelope["ok"], json!(false), "{envelope}");
    assert_eq!(envelope["error"]["code"], json!("backend_unavailable"));
    assert_eq!(envelope["error"]["retryable"], json!(true));
    assert_eq!(envelope["error"]["details"]["argument"], json!("--store"));
}

/// §4.1: `invalid_argument` is the one code carried at PROTOCOL level -- it rides
/// the JSON-RPC error and has no envelope at all. A caller distinguishes the two
/// levels without parsing prose.
#[test]
fn an_undeclared_parameter_is_a_protocol_error_with_no_envelope() {
    let session = Session::new("badparam").seeded(&notes());
    let responses = session.run(&[
        initialize(),
        json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "query", "arguments": {"q": example_query(), "cursor": "x"}}
        }),
    ]);

    let error = &responses[1]["error"];
    assert!(
        responses[1].get("result").is_none(),
        "a protocol-level refusal carries no result: {}",
        responses[1]
    );
    assert_eq!(error["code"], json!(-32602));
    assert_eq!(error["data"]["code"], json!("invalid_argument"));
    assert_eq!(error["data"]["details"]["parameter"], json!("cursor"));
}

/// The two notes the fixture seeds. The second exists so that a match on the
/// first is a MATCH and not "everything came back".
fn notes() -> Vec<SeedNote<'static>> {
    vec![
        SeedNote {
            keep_id: "keep-1",
            branch_id: BRANCH,
            content: "the walking skeleton crosses every seam it claims",
        },
        SeedNote {
            keep_id: "keep-2",
            branch_id: BRANCH,
            content: "a guard that arrives first is free",
        },
    ]
}

/// The query the tool description ships as its worked example.
fn example_query() -> String {
    query_for("skeleton")
}

fn query_for(term: &str) -> String {
    format!(
        "SELECT TEXT note FROM {} WHERE q MATCH '{term}'",
        Collection::Scratchpad.name()
    )
}

fn initialize() -> Value {
    json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"clientInfo": {"name": "wo013-test", "version": "0"}}
    })
}

/// One test's private directory, its store, and the session it drives.
struct Session {
    dir: PathBuf,
    store: Option<PathBuf>,
}

impl Session {
    fn new(label: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("wqm-wo013-{}-{label}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("the test directory is created");
        Session { dir, store: None }
    }

    /// Seed a store through the committed fixture -- the same schema and the same
    /// triggers the binary reads, rather than a second copy of either.
    fn seeded(mut self, notes: &[SeedNote<'_>]) -> Self {
        let path = self.dir.join("store.db");
        let _ = std::fs::remove_file(&path);
        let mut conn = at_path(&path).expect("the store is created with its schema");
        seed(&mut conn, &WriteTarget::of(Collection::Scratchpad), notes)
            .expect("the fixture seeds");
        self.store = Some(path);
        self
    }

    /// Run `initialize` then one `query`, and return the envelope.
    fn query(&self, q: &str) -> Value {
        let responses = self.run(&[
            initialize(),
            json!({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "query", "arguments": {"q": q}}
            }),
        ]);
        responses[1]["result"]["structuredContent"].clone()
    }

    /// Drive the real binary over a real pipe, one request per line.
    fn run(&self, requests: &[Value]) -> Vec<Value> {
        let mut command = Command::new(MCP_BIN);
        // A path nobody is serving. `query` must not need it -- reads are
        // in-process -- and this is where that stops being a claim.
        command
            .arg("--daemon-socket")
            .arg(self.dir.join("absent.sock"));
        if let Some(store) = &self.store {
            command.arg("--store").arg(store);
        }

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the MCP binary starts");

        {
            let stdin = child.stdin.as_mut().expect("stdin is piped");
            for request in requests {
                writeln!(stdin, "{request}").expect("the request is written");
            }
        }
        // Closing stdin ends the session, so the child exits on its own.
        drop(child.stdin.take());

        let output = child.wait_with_output().expect("the session ends");
        assert!(
            output.status.success(),
            "the server exited {:?}; stderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );

        let responses: Vec<Value> = BufReader::new(output.stdout.as_slice())
            .lines()
            .map(|line| line.expect("stdout is text"))
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(&line).expect("one JSON-RPC message per line"))
            .collect();
        assert_eq!(
            responses.len(),
            requests.len(),
            "one response per request: {responses:?}"
        );
        responses
    }
}

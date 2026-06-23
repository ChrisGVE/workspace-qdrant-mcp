// Guard 2 / AC-F1.6: QdrantReadClient exposes no mutating method.
// `upsert_points` does not exist on the read newtype, so this fails to compile.
use wqm_storage::QdrantReadClient;

fn assert_no_mutation(client: &QdrantReadClient) {
    let _ = client.upsert_points();
}

fn main() {}

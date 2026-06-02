//! DLQ (dead letter queue) CLI subcommands.

use anyhow::Result;
use clap::Subcommand;

use crate::grpc::DaemonClient;

#[derive(Subcommand)]
pub enum DlqCommand {
    /// List dead letter queue entries
    List {
        /// Filter by tenant ID
        #[arg(long)]
        tenant: Option<String>,
        /// Filter by error category
        #[arg(long)]
        category: Option<String>,
        /// Maximum entries to show
        #[arg(short, long, default_value = "50")]
        limit: i32,
    },
    /// Replay a DLQ item back into the queue
    Replay {
        /// DLQ entry ID
        id: String,
        /// Override permanent_data guard
        #[arg(long)]
        force: bool,
    },
    /// Purge expired DLQ entries
    Purge {
        /// Remove entries older than N days (default: 30)
        #[arg(long, default_value = "30")]
        older_than: i32,
    },
}

pub async fn execute(cmd: DlqCommand) -> Result<()> {
    let mut client = DaemonClient::connect_default().await?;
    match cmd {
        DlqCommand::List {
            tenant,
            category,
            limit,
        } => execute_list(&mut client, tenant, category, limit).await,
        DlqCommand::Replay { id, force } => execute_replay(&mut client, &id, force).await,
        DlqCommand::Purge { older_than } => execute_purge(&mut client, older_than).await,
    }
}

async fn execute_list(
    client: &mut DaemonClient,
    tenant: Option<String>,
    category: Option<String>,
    limit: i32,
) -> Result<()> {
    let response = client
        .system()
        .list_dlq(crate::grpc::proto::ListDlqRequest {
            tenant_id: tenant.unwrap_or_default(),
            category: category.unwrap_or_default(),
            limit,
            offset: 0,
        })
        .await?
        .into_inner();

    if response.entries.is_empty() {
        println!("Dead letter queue is empty.");
        return Ok(());
    }

    println!(
        "{:<34} {:<10} {:<20} {:<20} ERROR",
        "DLQ_ID", "TYPE", "CATEGORY", "MOVED_AT"
    );
    println!("{}", "-".repeat(110));
    for entry in &response.entries {
        let error_preview = if entry.error_message.len() > 40 {
            format!("{}...", &entry.error_message[..40])
        } else {
            entry.error_message.clone()
        };
        println!(
            "{:<34} {:<10} {:<20} {:<20} {}",
            &entry.dlq_id[..entry.dlq_id.len().min(32)],
            entry.item_type,
            entry.error_category,
            &entry.moved_to_dlq_at[..entry.moved_to_dlq_at.len().min(19)],
            error_preview,
        );
    }
    println!("\nTotal: {} entries", response.total);
    Ok(())
}

async fn execute_replay(client: &mut DaemonClient, dlq_id: &str, force: bool) -> Result<()> {
    let response = client
        .queue_write()
        .replay_dlq_item(crate::grpc::proto::ReplayDlqItemRequest {
            dlq_id: dlq_id.to_string(),
            force,
        })
        .await?
        .into_inner();

    if response.success {
        println!(
            "Replayed DLQ item → new queue ID: {}",
            response.new_queue_id
        );
    } else {
        eprintln!("Replay failed: {}", response.error);
        std::process::exit(1);
    }
    Ok(())
}

async fn execute_purge(client: &mut DaemonClient, older_than: i32) -> Result<()> {
    let mut total_deleted = 0i32;

    loop {
        let response = client
            .queue_write()
            .purge_dlq(crate::grpc::proto::PurgeDlqRequest {
                retention_days: older_than,
                tenant_id: String::new(),
                category: String::new(),
            })
            .await?
            .into_inner();

        total_deleted += response.rows_deleted;
        if !response.has_more {
            break;
        }
    }

    println!(
        "Purged {} DLQ entries older than {} days",
        total_deleted, older_than
    );
    Ok(())
}

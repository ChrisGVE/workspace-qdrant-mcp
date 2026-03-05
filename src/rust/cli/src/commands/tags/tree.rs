//! Tree subcommand handler: `tags tree` — displays the canonical tag hierarchy.

use anyhow::Result;

use super::db::{open_db, table_exists};
use crate::output;

struct TreeNode {
    id: i64,
    name: String,
    level: i32,
    parent_id: Option<i64>,
}

pub(super) fn show_tree(tenant_id: &str, collection: &str) -> Result<()> {
    let conn = open_db()?;
    if !table_exists(&conn, "canonical_tags") {
        anyhow::bail!("Canonical tags table not found. Ensure daemon schema v16+ is applied.");
    }

    let mut stmt = conn.prepare(
        "SELECT canonical_id, canonical_name, level, parent_id \
         FROM canonical_tags \
         WHERE tenant_id = ? AND collection = ? \
         ORDER BY level ASC, canonical_name ASC",
    )?;

    let nodes: Vec<TreeNode> = stmt
        .query_map(rusqlite::params![tenant_id, collection], |row| {
            Ok(TreeNode {
                id: row.get(0)?,
                name: row.get(1)?,
                level: row.get(2)?,
                parent_id: row.get(3)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if nodes.is_empty() {
        output::info(format!(
            "No canonical tag hierarchy found for tenant {}. Run 'wqm tags rebuild --tenant {}'",
            tenant_id, tenant_id
        ));
        return Ok(());
    }

    output::info(format!(
        "Canonical tag hierarchy for tenant {} ({} tags)",
        tenant_id,
        nodes.len()
    ));
    println!();

    let roots: Vec<&TreeNode> = nodes.iter().filter(|n| n.level == 1).collect();
    for (i, root) in roots.iter().enumerate() {
        let is_last_root = i == roots.len() - 1;
        let prefix = if is_last_root {
            "└── "
        } else {
            "├── "
        };
        println!("{}{} (L1)", prefix, root.name);

        let children: Vec<&TreeNode> = nodes
            .iter()
            .filter(|n| n.level == 2 && n.parent_id == Some(root.id))
            .collect();

        for (j, child) in children.iter().enumerate() {
            let is_last_child = j == children.len() - 1;
            let indent = if is_last_root { "    " } else { "│   " };
            let child_prefix = if is_last_child {
                "└── "
            } else {
                "├── "
            };
            println!("{}{}{} (L2)", indent, child_prefix, child.name);

            let grandchildren: Vec<&TreeNode> = nodes
                .iter()
                .filter(|n| n.level == 3 && n.parent_id == Some(child.id))
                .collect();

            for (k, gc) in grandchildren.iter().enumerate() {
                let is_last_gc = k == grandchildren.len() - 1;
                let gc_indent = if is_last_root { "    " } else { "│   " };
                let gc_indent2 = if is_last_child { "    " } else { "│   " };
                let gc_prefix = if is_last_gc {
                    "└── "
                } else {
                    "├── "
                };
                println!("{}{}{}{}", gc_indent, gc_indent2, gc_prefix, gc.name);
            }
        }
    }

    let orphan_l2: Vec<&TreeNode> = nodes
        .iter()
        .filter(|n| n.level == 2 && n.parent_id.is_none())
        .collect();
    if !orphan_l2.is_empty() {
        println!();
        output::info(format!("Unlinked level 2 tags ({})", orphan_l2.len()));
        for n in &orphan_l2 {
            println!("  - {} (L2)", n.name);
        }
    }

    let orphan_l3: Vec<&TreeNode> = nodes
        .iter()
        .filter(|n| n.level == 3 && n.parent_id.is_none())
        .collect();
    if !orphan_l3.is_empty() {
        println!();
        output::info(format!("Unlinked level 3 tags ({})", orphan_l3.len()));
        for n in &orphan_l3 {
            println!("  - {}", n.name);
        }
    }

    Ok(())
}

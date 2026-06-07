//! `wqm groups list` -- show all project group memberships.

use anyhow::Result;
use tabled::Tabled;

use crate::data::db::{connect_readonly, table_exists};
use crate::output;

#[derive(Tabled, serde::Serialize)]
pub(super) struct GroupRow {
    #[tabled(rename = "Group ID")]
    pub group_id: String,
    #[tabled(rename = "Project")]
    pub project: String,
    #[tabled(rename = "Tenant ID")]
    pub tenant_id: String,
    #[tabled(rename = "Type")]
    pub group_type: String,
    #[tabled(rename = "Confidence")]
    pub confidence: String,
}

pub(super) fn list_groups(
    tenant: Option<&str>,
    group_type: Option<&str>,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
    let conn = connect_readonly()?;

    if !table_exists(&conn, "project_groups") {
        anyhow::bail!("project_groups table not found. Ensure daemon schema v24+ is applied.");
    }

    // Build query dynamically based on filters
    let mut sql =
        String::from("SELECT group_id, tenant_id, group_type, confidence FROM project_groups");
    let mut conditions: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(t) = tenant {
        conditions.push(format!("tenant_id = ?{}", params.len() + 1));
        params.push(Box::new(t.to_string()));
    }
    if let Some(gt) = group_type {
        conditions.push(format!("group_type = ?{}", params.len() + 1));
        params.push(Box::new(gt.to_string()));
    }

    if !conditions.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&conditions.join(" AND "));
    }

    sql.push_str(" ORDER BY group_type, group_id, tenant_id");

    let mut stmt = conn.prepare(&sql)?;
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let names = crate::data::tenants::name_map_in(&conn);
    let rows: Vec<GroupRow> = stmt
        .query_map(params_refs.as_slice(), |row| {
            let confidence: f64 = row.get(3)?;
            let tenant_id: String = row.get(1)?;
            Ok(GroupRow {
                group_id: row.get(0)?,
                project: crate::data::tenants::display_name(&names, &tenant_id),
                tenant_id,
                group_type: row.get(2)?,
                confidence: format!("{:.2}", confidence),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if rows.is_empty() {
        let scope = match (tenant, group_type) {
            (Some(t), Some(gt)) => format!("tenant {} with type {}", t, gt),
            (Some(t), None) => format!("tenant {}", t),
            (None, Some(gt)) => format!("type {}", gt),
            (None, None) => "any project".to_string(),
        };
        output::info(format!("No group memberships found for {}", scope));
        return Ok(());
    }

    if json {
        output::print_json(&rows);
    } else if script {
        output::print_script(&rows, !no_headers);
    } else {
        let count = rows.len();
        output::print_table(&rows);
        output::summary(output::summary_line(count, count, "group memberships"));
    }

    Ok(())
}

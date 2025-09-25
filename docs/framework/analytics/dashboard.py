"""Analytics dashboard for visualizing documentation usage patterns."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from .storage import AnalyticsStorage, AnalyticsStats
from .privacy import PrivacyManager, ConsentLevel


logger = logging.getLogger(__name__)


@dataclass
class DashboardMetric:
    """Represents a single dashboard metric."""
    name: str
    value: Any
    unit: str
    change_percent: Optional[float] = None
    trend: Optional[str] = None  # 'up', 'down', 'stable'


@dataclass
class ChartData:
    """Data structure for charts."""
    chart_type: str  # 'line', 'bar', 'pie', etc.
    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = None


@dataclass
class DashboardData:
    """Complete dashboard data structure."""
    generated_at: datetime
    date_range: Tuple[datetime, datetime]
    summary_metrics: List[DashboardMetric]
    charts: List[ChartData]
    tables: List[Dict[str, Any]]
    insights: List[str]


class AnalyticsDashboard:
    """Generates dashboard data and reports from analytics storage."""

    def __init__(self, storage: AnalyticsStorage, privacy_manager: PrivacyManager = None):
        """Initialize analytics dashboard.

        Args:
            storage: Analytics storage backend
            privacy_manager: Privacy manager for access control
        """
        self.storage = storage
        self.privacy_manager = privacy_manager

    def generate_dashboard(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        consent_level: Optional[ConsentLevel] = None
    ) -> Optional[DashboardData]:
        """Generate complete dashboard data.

        Args:
            start_date: Start date for analytics period
            end_date: End date for analytics period
            consent_level: User consent level to respect

        Returns:
            Dashboard data or None if generation failed
        """
        try:
            # Set default date range to last 30 days
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)

            # Check privacy constraints
            effective_consent = consent_level or (
                self.privacy_manager.get_consent_level() if self.privacy_manager else ConsentLevel.ALL
            )

            # Get basic stats
            stats = self.storage.get_stats(start_date, end_date)
            if not stats:
                logger.warning("Failed to get analytics stats for dashboard")
                return None

            # Generate metrics
            summary_metrics = self._generate_summary_metrics(stats, start_date, end_date)

            # Generate charts (respecting privacy)
            charts = self._generate_charts(start_date, end_date, effective_consent)

            # Generate tables
            tables = self._generate_tables(start_date, end_date, effective_consent)

            # Generate insights
            insights = self._generate_insights(stats, start_date, end_date)

            return DashboardData(
                generated_at=datetime.now(),
                date_range=(start_date, end_date),
                summary_metrics=summary_metrics,
                charts=charts,
                tables=tables,
                insights=insights
            )

        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            return None

    def _generate_summary_metrics(
        self,
        stats: AnalyticsStats,
        start_date: datetime,
        end_date: datetime
    ) -> List[DashboardMetric]:
        """Generate summary metrics for the dashboard.

        Args:
            stats: Analytics statistics
            start_date: Period start date
            end_date: Period end date

        Returns:
            List of dashboard metrics
        """
        metrics = []

        # Total page views
        metrics.append(DashboardMetric(
            name="Total Page Views",
            value=stats.total_page_views,
            unit="views",
            trend=self._calculate_trend(stats.total_page_views, start_date, end_date, 'page_view')
        ))

        # Unique sessions
        metrics.append(DashboardMetric(
            name="Unique Sessions",
            value=stats.unique_sessions,
            unit="sessions",
            trend=self._calculate_trend(stats.unique_sessions, start_date, end_date, 'session')
        ))

        # Average session duration
        avg_duration_minutes = stats.avg_session_duration_ms / (1000 * 60)
        metrics.append(DashboardMetric(
            name="Avg. Session Duration",
            value=f"{avg_duration_minutes:.1f}",
            unit="minutes"
        ))

        # Error rate
        metrics.append(DashboardMetric(
            name="Error Rate",
            value=f"{stats.error_rate:.1f}",
            unit="%",
            trend="down" if stats.error_rate < 5.0 else "up"
        ))

        # Top search queries count
        metrics.append(DashboardMetric(
            name="Search Queries",
            value=len(stats.top_search_queries),
            unit="unique queries"
        ))

        return metrics

    def _calculate_trend(
        self,
        current_value: int,
        start_date: datetime,
        end_date: datetime,
        event_type: str
    ) -> Optional[str]:
        """Calculate trend for a metric compared to previous period.

        Args:
            current_value: Current period value
            start_date: Current period start
            end_date: Current period end
            event_type: Type of events to compare

        Returns:
            Trend direction ('up', 'down', 'stable') or None
        """
        try:
            # Calculate previous period
            period_length = end_date - start_date
            prev_start = start_date - period_length
            prev_end = start_date

            # Get previous period stats
            prev_stats = self.storage.get_stats(prev_start, prev_end)
            if not prev_stats:
                return None

            if event_type == 'page_view':
                prev_value = prev_stats.total_page_views
            elif event_type == 'session':
                prev_value = prev_stats.unique_sessions
            else:
                return None

            if prev_value == 0:
                return "up" if current_value > 0 else "stable"

            change_percent = ((current_value - prev_value) / prev_value) * 100

            if change_percent > 5:
                return "up"
            elif change_percent < -5:
                return "down"
            else:
                return "stable"

        except Exception as e:
            logger.warning(f"Failed to calculate trend: {e}")
            return None

    def _generate_charts(
        self,
        start_date: datetime,
        end_date: datetime,
        consent_level: ConsentLevel
    ) -> List[ChartData]:
        """Generate chart data for the dashboard.

        Args:
            start_date: Period start date
            end_date: Period end date
            consent_level: User consent level

        Returns:
            List of chart data
        """
        charts = []

        try:
            # Page views over time (allowed for FUNCTIONAL and above)
            if consent_level.value in ['functional', 'analytics', 'all']:
                page_views_chart = self._generate_page_views_chart(start_date, end_date)
                if page_views_chart:
                    charts.append(page_views_chart)

            # Top pages chart (allowed for FUNCTIONAL and above)
            if consent_level.value in ['functional', 'analytics', 'all']:
                top_pages_chart = self._generate_top_pages_chart(start_date, end_date)
                if top_pages_chart:
                    charts.append(top_pages_chart)

            # Search queries chart (allowed for ANALYTICS and above)
            if consent_level.value in ['analytics', 'all']:
                search_chart = self._generate_search_queries_chart(start_date, end_date)
                if search_chart:
                    charts.append(search_chart)

            # Error types chart (allowed for ESSENTIAL and above)
            error_chart = self._generate_error_types_chart(start_date, end_date)
            if error_chart:
                charts.append(error_chart)

        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")

        return charts

    def _generate_page_views_chart(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[ChartData]:
        """Generate page views over time chart.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            Chart data or None if failed
        """
        try:
            # Get daily page views
            daily_views = self._get_daily_page_views(start_date, end_date)

            if not daily_views:
                return None

            labels = []
            data = []

            current_date = start_date.date()
            end_date_only = end_date.date()

            while current_date <= end_date_only:
                labels.append(current_date.strftime("%Y-%m-%d"))
                data.append(daily_views.get(current_date, 0))
                current_date += timedelta(days=1)

            return ChartData(
                chart_type="line",
                title="Page Views Over Time",
                labels=labels,
                datasets=[{
                    "label": "Page Views",
                    "data": data,
                    "borderColor": "rgb(59, 130, 246)",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "fill": True
                }],
                options={
                    "responsive": True,
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": "Page Views"
                            }
                        }
                    }
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate page views chart: {e}")
            return None

    def _get_daily_page_views(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[datetime, int]:
        """Get daily page view counts.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            Dictionary mapping dates to page view counts
        """
        try:
            events = self.storage.get_events(
                start_date=start_date,
                end_date=end_date,
                event_type='page_view',
                limit=10000
            )

            daily_counts = {}
            for event in events:
                event_date = event.timestamp.date()
                daily_counts[event_date] = daily_counts.get(event_date, 0) + 1

            return daily_counts

        except Exception as e:
            logger.error(f"Failed to get daily page views: {e}")
            return {}

    def _generate_top_pages_chart(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[ChartData]:
        """Generate top pages chart.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            Chart data or None if failed
        """
        try:
            stats = self.storage.get_stats(start_date, end_date)
            if not stats or not stats.top_pages:
                return None

            # Take top 10 pages
            top_pages = stats.top_pages[:10]

            labels = [page['page'] for page in top_pages]
            data = [page['views'] for page in top_pages]

            return ChartData(
                chart_type="bar",
                title="Top Pages",
                labels=labels,
                datasets=[{
                    "label": "Page Views",
                    "data": data,
                    "backgroundColor": [
                        "rgba(59, 130, 246, 0.8)",
                        "rgba(16, 185, 129, 0.8)",
                        "rgba(245, 101, 101, 0.8)",
                        "rgba(251, 191, 36, 0.8)",
                        "rgba(139, 92, 246, 0.8)",
                        "rgba(236, 72, 153, 0.8)",
                        "rgba(6, 182, 212, 0.8)",
                        "rgba(34, 197, 94, 0.8)",
                        "rgba(249, 115, 22, 0.8)",
                        "rgba(168, 85, 247, 0.8)"
                    ]
                }],
                options={
                    "responsive": True,
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": "Views"
                            }
                        }
                    }
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate top pages chart: {e}")
            return None

    def _generate_search_queries_chart(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[ChartData]:
        """Generate search queries chart.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            Chart data or None if failed
        """
        try:
            stats = self.storage.get_stats(start_date, end_date)
            if not stats or not stats.top_search_queries:
                return None

            # Take top 8 search queries
            top_queries = stats.top_search_queries[:8]

            labels = [query['query'] for query in top_queries]
            data = [query['count'] for query in top_queries]

            return ChartData(
                chart_type="doughnut",
                title="Top Search Queries",
                labels=labels,
                datasets=[{
                    "data": data,
                    "backgroundColor": [
                        "rgba(59, 130, 246, 0.8)",
                        "rgba(16, 185, 129, 0.8)",
                        "rgba(245, 101, 101, 0.8)",
                        "rgba(251, 191, 36, 0.8)",
                        "rgba(139, 92, 246, 0.8)",
                        "rgba(236, 72, 153, 0.8)",
                        "rgba(6, 182, 212, 0.8)",
                        "rgba(34, 197, 94, 0.8)"
                    ]
                }],
                options={
                    "responsive": True,
                    "plugins": {
                        "legend": {
                            "position": "bottom"
                        }
                    }
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate search queries chart: {e}")
            return None

    def _generate_error_types_chart(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[ChartData]:
        """Generate error types chart.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            Chart data or None if failed
        """
        try:
            error_events = self.storage.get_events(
                start_date=start_date,
                end_date=end_date,
                event_type='error',
                limit=1000
            )

            if not error_events:
                return None

            # Count error types
            error_counts = {}
            for event in error_events:
                if event.metadata and 'error_type' in event.metadata:
                    error_type = event.metadata['error_type']
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1

            if not error_counts:
                return None

            # Sort by count and take top 10
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            labels = [error[0] for error in sorted_errors]
            data = [error[1] for error in sorted_errors]

            return ChartData(
                chart_type="bar",
                title="Error Types",
                labels=labels,
                datasets=[{
                    "label": "Error Count",
                    "data": data,
                    "backgroundColor": "rgba(245, 101, 101, 0.8)",
                    "borderColor": "rgba(245, 101, 101, 1)",
                    "borderWidth": 1
                }],
                options={
                    "responsive": True,
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": "Count"
                            }
                        }
                    }
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate error types chart: {e}")
            return None

    def _generate_tables(
        self,
        start_date: datetime,
        end_date: datetime,
        consent_level: ConsentLevel
    ) -> List[Dict[str, Any]]:
        """Generate table data for the dashboard.

        Args:
            start_date: Period start date
            end_date: Period end date
            consent_level: User consent level

        Returns:
            List of table data
        """
        tables = []

        try:
            # Recent errors table (allowed for ESSENTIAL and above)
            errors_table = self._generate_recent_errors_table(start_date, end_date)
            if errors_table:
                tables.append(errors_table)

            # Session details table (allowed for ANALYTICS and above)
            if consent_level.value in ['analytics', 'all']:
                sessions_table = self._generate_sessions_table(start_date, end_date)
                if sessions_table:
                    tables.append(sessions_table)

        except Exception as e:
            logger.error(f"Failed to generate tables: {e}")

        return tables

    def _generate_recent_errors_table(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Generate recent errors table.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            Table data or None if failed
        """
        try:
            error_events = self.storage.get_events(
                start_date=start_date,
                end_date=end_date,
                event_type='error',
                limit=50
            )

            if not error_events:
                return None

            rows = []
            for event in error_events:
                rows.append({
                    'timestamp': event.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'page': event.page_path,
                    'error_type': event.metadata.get('error_type', 'Unknown') if event.metadata else 'Unknown',
                    'error_message': event.metadata.get('error_message', '')[:100] + '...' if event.metadata and event.metadata.get('error_message', '') else 'No message'
                })

            return {
                'title': 'Recent Errors',
                'columns': [
                    {'key': 'timestamp', 'label': 'Time'},
                    {'key': 'page', 'label': 'Page'},
                    {'key': 'error_type', 'label': 'Type'},
                    {'key': 'error_message', 'label': 'Message'}
                ],
                'rows': rows
            }

        except Exception as e:
            logger.error(f"Failed to generate recent errors table: {e}")
            return None

    def _generate_sessions_table(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Generate sessions summary table.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            Table data or None if failed
        """
        try:
            # Get session events
            events = self.storage.get_events(
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )

            if not events:
                return None

            # Group events by session
            sessions = {}
            for event in events:
                session_id = event.session_id
                if session_id not in sessions:
                    sessions[session_id] = {
                        'session_id': session_id,
                        'first_seen': event.timestamp,
                        'last_seen': event.timestamp,
                        'page_views': 0,
                        'total_duration': 0,
                        'pages_visited': set()
                    }

                session = sessions[session_id]
                if event.timestamp < session['first_seen']:
                    session['first_seen'] = event.timestamp
                if event.timestamp > session['last_seen']:
                    session['last_seen'] = event.timestamp

                if event.event_type == 'page_view':
                    session['page_views'] += 1
                    session['pages_visited'].add(event.page_path)

                if event.duration_ms:
                    session['total_duration'] += event.duration_ms

            # Convert to table rows
            rows = []
            for session_id, session_data in list(sessions.items())[:20]:  # Top 20 sessions
                duration_minutes = session_data['total_duration'] / (1000 * 60)
                session_length = (session_data['last_seen'] - session_data['first_seen']).total_seconds() / 60

                rows.append({
                    'session_id': session_id[:8] + '...',  # Truncate for display
                    'first_seen': session_data['first_seen'].strftime('%Y-%m-%d %H:%M'),
                    'duration_minutes': f"{duration_minutes:.1f}",
                    'session_length_minutes': f"{session_length:.1f}",
                    'page_views': session_data['page_views'],
                    'unique_pages': len(session_data['pages_visited'])
                })

            return {
                'title': 'Recent Sessions',
                'columns': [
                    {'key': 'session_id', 'label': 'Session ID'},
                    {'key': 'first_seen', 'label': 'Started'},
                    {'key': 'duration_minutes', 'label': 'Active Time (min)'},
                    {'key': 'session_length_minutes', 'label': 'Session Length (min)'},
                    {'key': 'page_views', 'label': 'Page Views'},
                    {'key': 'unique_pages', 'label': 'Unique Pages'}
                ],
                'rows': rows
            }

        except Exception as e:
            logger.error(f"Failed to generate sessions table: {e}")
            return None

    def _generate_insights(
        self,
        stats: AnalyticsStats,
        start_date: datetime,
        end_date: datetime
    ) -> List[str]:
        """Generate insights based on analytics data.

        Args:
            stats: Analytics statistics
            start_date: Period start date
            end_date: Period end date

        Returns:
            List of insight strings
        """
        insights = []

        try:
            # Session duration insight
            avg_duration_minutes = stats.avg_session_duration_ms / (1000 * 60)
            if avg_duration_minutes > 10:
                insights.append("Users are spending significant time on documentation (avg {:.1f} minutes per session), indicating high engagement.".format(avg_duration_minutes))
            elif avg_duration_minutes < 2:
                insights.append("Short average session duration ({:.1f} minutes) may indicate users are not finding what they need quickly.".format(avg_duration_minutes))

            # Error rate insight
            if stats.error_rate > 5:
                insights.append("High error rate ({:.1f}%) detected. Consider reviewing error logs to improve user experience.".format(stats.error_rate))
            elif stats.error_rate < 1:
                insights.append("Low error rate ({:.1f}%) indicates good documentation stability.".format(stats.error_rate))

            # Popular content insight
            if stats.top_pages:
                top_page = stats.top_pages[0]
                total_views = sum(page['views'] for page in stats.top_pages[:5])
                if top_page['views'] > total_views * 0.5:
                    insights.append("The page '{}' dominates traffic ({} views), consider creating more entry points to other content.".format(
                        top_page['page'], top_page['views']
                    ))

            # Search behavior insight
            if stats.top_search_queries:
                total_searches = sum(query['count'] for query in stats.top_search_queries)
                if total_searches > stats.total_page_views * 0.3:
                    insights.append("High search usage ({} searches vs {} page views) suggests users need better navigation or content discovery.".format(
                        total_searches, stats.total_page_views
                    ))

            # Session engagement insight
            if stats.unique_sessions > 0:
                pages_per_session = stats.total_page_views / stats.unique_sessions
                if pages_per_session > 5:
                    insights.append("High page views per session ({:.1f}) indicates users are exploring content thoroughly.".format(pages_per_session))
                elif pages_per_session < 2:
                    insights.append("Low page views per session ({:.1f}) may indicate poor content discoverability or user experience issues.".format(pages_per_session))

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")

        return insights

    def export_dashboard_data(
        self,
        output_path: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """Export dashboard data to JSON file.

        Args:
            output_path: Path for exported dashboard data
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            True if export successful, False otherwise
        """
        try:
            dashboard_data = self.generate_dashboard(start_date, end_date)
            if not dashboard_data:
                return False

            # Convert to serializable format
            export_data = {
                'generated_at': dashboard_data.generated_at.isoformat(),
                'date_range': {
                    'start': dashboard_data.date_range[0].isoformat(),
                    'end': dashboard_data.date_range[1].isoformat()
                },
                'summary_metrics': [asdict(metric) for metric in dashboard_data.summary_metrics],
                'charts': [asdict(chart) for chart in dashboard_data.charts],
                'tables': dashboard_data.tables,
                'insights': dashboard_data.insights
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Dashboard data exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export dashboard data: {e}")
            return False
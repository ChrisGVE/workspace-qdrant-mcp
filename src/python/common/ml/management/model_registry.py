"""
Model Registry System for ML Model Management

Provides comprehensive model versioning, storage, retrieval, and metadata management
with support for model lineage tracking, performance comparison, and automated
deployment workflows with extensive validation and error handling.
"""

import hashlib
import json
import logging
import pickle
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config.ml_config import MLConfig
from ..pipeline.training_pipeline import TrainingResult


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    name: str
    version: str
    description: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: dict[str, str] = field(default_factory=dict)
    stage: str = "development"  # development, staging, production, archived
    model_type: str = ""
    task_type: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    model_size_bytes: int = 0
    model_hash: str = ""
    parent_run_id: str | None = None
    experiment_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        # Convert ISO format strings back to datetime objects
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    model_id: str
    version: str
    metadata: ModelMetadata
    model_path: Path
    artifacts_path: Path | None = None


class ModelRegistryError(Exception):
    """Base exception for model registry operations."""
    pass


class ModelNotFoundError(ModelRegistryError):
    """Exception raised when a model is not found."""
    pass


class ModelVersionError(ModelRegistryError):
    """Exception raised for version-related errors."""
    pass


class ModelStorageError(ModelRegistryError):
    """Exception raised for storage operations."""
    pass


class ModelRegistry:
    """
    Comprehensive model registry for ML model lifecycle management.

    Features:
    - Model versioning with semantic versioning support
    - Metadata tracking and search capabilities
    - Model lineage and experiment tracking
    - Performance metrics storage and comparison
    - Stage-based deployment workflow (dev -> staging -> prod)
    - Model archival and cleanup
    - Concurrent access safety with database transactions
    - Model validation and integrity checking
    """

    def __init__(self, config: MLConfig):
        """
        Initialize model registry.

        Args:
            config: ML configuration containing registry settings

        Raises:
            ModelRegistryError: If registry initialization fails
        """
        self.config = config
        self.registry_path = config.model_directory / "registry"
        self.models_path = config.model_directory / "models"
        self.db_path = self.registry_path / "models.db"

        # Create directories
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()

        try:
            self._initialize_database()
        except Exception as e:
            raise ModelRegistryError(f"Failed to initialize model registry: {str(e)}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for model registry."""
        logger = logging.getLogger("model_registry")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_database(self) -> None:
        """Initialize SQLite database for metadata storage."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()

            # Create models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT 'development',
                    model_type TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    model_size_bytes INTEGER DEFAULT 0,
                    model_hash TEXT,
                    parent_run_id TEXT,
                    experiment_name TEXT,
                    UNIQUE(name, version)
                )
            """)

            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    model_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    PRIMARY KEY (model_id, metric_name),
                    FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE
                )
            """)

            # Create hyperparameters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_hyperparameters (
                    model_id TEXT,
                    param_name TEXT,
                    param_value TEXT,
                    PRIMARY KEY (model_id, param_name),
                    FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE
                )
            """)

            # Create tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_tags (
                    model_id TEXT,
                    tag_key TEXT,
                    tag_value TEXT,
                    PRIMARY KEY (model_id, tag_key),
                    FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE
                )
            """)

            # Create feature names table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_features (
                    model_id TEXT,
                    feature_name TEXT,
                    feature_index INTEGER,
                    PRIMARY KEY (model_id, feature_index),
                    FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_name ON models (name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_stage ON models (stage)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_created_at ON models (created_at)")

            conn.commit()

    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                isolation_level=None  # Autocommit mode
            )
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise ModelStorageError(f"Database operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    def register_model(
        self,
        training_result: TrainingResult,
        name: str,
        version: str | None = None,
        description: str | None = None,
        tags: dict[str, str] | None = None,
        stage: str = "development",
        experiment_name: str | None = None
    ) -> str:
        """
        Register a trained model in the registry.

        Args:
            training_result: Result from training pipeline
            name: Model name
            version: Model version (auto-generated if not provided)
            description: Model description
            tags: Additional tags for the model
            stage: Deployment stage
            experiment_name: Associated experiment name

        Returns:
            Model ID of registered model

        Raises:
            ModelRegistryError: If registration fails
            ModelVersionError: If version already exists
        """
        try:
            # Generate version if not provided
            if not version:
                version = self._generate_next_version(name)

            # Generate model ID
            model_id = f"{name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Store model artifacts
            model_path = self._store_model_artifacts(model_id, training_result)

            # Calculate model hash for integrity checking
            model_hash = self._calculate_model_hash(model_path)

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                description=description or f"Model {name} version {version}",
                tags=tags or {},
                stage=stage,
                model_type=training_result.model_config.model_type,
                task_type=training_result.model_config.task_type,
                metrics=training_result.metrics.to_dict(),
                hyperparameters=training_result.best_params or {},
                feature_names=training_result.feature_names,
                model_size_bytes=model_path.stat().st_size,
                model_hash=model_hash,
                experiment_name=experiment_name
            )

            # Store metadata in database
            self._store_metadata(metadata)

            self.logger.info(f"Model registered successfully: {model_id}")
            return model_id

        except Exception as e:
            # Clean up partially stored artifacts
            try:
                self._cleanup_model_artifacts(model_id)
            except:
                pass
            raise ModelRegistryError(f"Failed to register model: {str(e)}")

    def get_model(self, model_id: str) -> ModelVersion:
        """
        Retrieve a model by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelVersion object

        Raises:
            ModelNotFoundError: If model not found
        """
        try:
            metadata = self._get_metadata(model_id)
            if not metadata:
                raise ModelNotFoundError(f"Model not found: {model_id}")

            model_path = self.models_path / model_id / "model.pkl"
            artifacts_path = self.models_path / model_id / "artifacts"

            if not model_path.exists():
                raise ModelStorageError(f"Model file not found: {model_path}")

            return ModelVersion(
                model_id=model_id,
                version=metadata.version,
                metadata=metadata,
                model_path=model_path,
                artifacts_path=artifacts_path if artifacts_path.exists() else None
            )

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelRegistryError(f"Failed to retrieve model {model_id}: {str(e)}")

    def get_model_by_name(
        self,
        name: str,
        version: str | None = None,
        stage: str | None = None
    ) -> ModelVersion:
        """
        Retrieve a model by name and optional version/stage.

        Args:
            name: Model name
            version: Specific version (latest if not provided)
            stage: Deployment stage filter

        Returns:
            ModelVersion object

        Raises:
            ModelNotFoundError: If model not found
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                if version:
                    # Get specific version
                    query = "SELECT model_id FROM models WHERE name = ? AND version = ?"
                    params = [name, version]

                    if stage:
                        query += " AND stage = ?"
                        params.append(stage)

                    cursor.execute(query, params)
                else:
                    # Get latest version
                    query = """
                        SELECT model_id FROM models
                        WHERE name = ?
                    """
                    params = [name]

                    if stage:
                        query += " AND stage = ?"
                        params.append(stage)

                    query += " ORDER BY created_at DESC LIMIT 1"
                    cursor.execute(query, params)

                row = cursor.fetchone()
                if not row:
                    stage_str = f" in stage {stage}" if stage else ""
                    version_str = f" version {version}" if version else ""
                    raise ModelNotFoundError(
                        f"Model {name}{version_str}{stage_str} not found"
                    )

                return self.get_model(row['model_id'])

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelRegistryError(f"Failed to retrieve model {name}: {str(e)}")

    def list_models(
        self,
        name_pattern: str | None = None,
        stage: str | None = None,
        tags: dict[str, str] | None = None,
        limit: int = 100
    ) -> list[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            name_pattern: Name pattern for filtering (SQL LIKE pattern)
            stage: Stage filter
            tags: Tag filters
            limit: Maximum number of results

        Returns:
            List of model metadata

        Raises:
            ModelRegistryError: If listing fails
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                query = "SELECT model_id FROM models WHERE 1=1"
                params = []

                if name_pattern:
                    query += " AND name LIKE ?"
                    params.append(name_pattern)

                if stage:
                    query += " AND stage = ?"
                    params.append(stage)

                # Add tag filtering if specified
                if tags:
                    for key, value in tags.items():
                        query += """
                            AND model_id IN (
                                SELECT model_id FROM model_tags
                                WHERE tag_key = ? AND tag_value = ?
                            )
                        """
                        params.extend([key, value])

                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                models = []
                for row in rows:
                    metadata = self._get_metadata(row['model_id'])
                    if metadata:
                        models.append(metadata)

                return models

        except Exception as e:
            raise ModelRegistryError(f"Failed to list models: {str(e)}")

    def update_model_stage(self, model_id: str, stage: str) -> None:
        """
        Update model deployment stage.

        Args:
            model_id: Model identifier
            stage: New stage (development, staging, production, archived)

        Raises:
            ModelNotFoundError: If model not found
            ModelRegistryError: If update fails
        """
        valid_stages = {"development", "staging", "production", "archived"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

        try:
            # Verify model exists
            if not self._get_metadata(model_id):
                raise ModelNotFoundError(f"Model not found: {model_id}")

            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE models SET stage = ?, updated_at = ? WHERE model_id = ?",
                    (stage, datetime.now().isoformat(), model_id)
                )

                if cursor.rowcount == 0:
                    raise ModelNotFoundError(f"Model not found: {model_id}")

                conn.commit()

            self.logger.info(f"Model {model_id} stage updated to {stage}")

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelRegistryError(f"Failed to update model stage: {str(e)}")

    def delete_model(self, model_id: str, force: bool = False) -> None:
        """
        Delete a model from the registry.

        Args:
            model_id: Model identifier
            force: Force deletion even if in production stage

        Raises:
            ModelNotFoundError: If model not found
            ModelRegistryError: If deletion fails or model is in production
        """
        try:
            # Get model metadata
            metadata = self._get_metadata(model_id)
            if not metadata:
                raise ModelNotFoundError(f"Model not found: {model_id}")

            # Check if model is in production
            if metadata.stage == "production" and not force:
                raise ModelRegistryError(
                    f"Cannot delete production model {model_id}. Use force=True to override."
                )

            # Delete from database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                conn.commit()

            # Delete model artifacts
            self._cleanup_model_artifacts(model_id)

            self.logger.info(f"Model {model_id} deleted successfully")

        except (ModelNotFoundError, ModelRegistryError):
            raise
        except Exception as e:
            raise ModelRegistryError(f"Failed to delete model {model_id}: {str(e)}")

    def compare_models(
        self,
        model_ids: list[str],
        metrics: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Compare multiple models by their metrics and metadata.

        Args:
            model_ids: List of model identifiers
            metrics: Specific metrics to compare (all if not provided)

        Returns:
            Dictionary with model comparison data

        Raises:
            ModelNotFoundError: If any model not found
            ModelRegistryError: If comparison fails
        """
        try:
            comparison = {}

            for model_id in model_ids:
                metadata = self._get_metadata(model_id)
                if not metadata:
                    raise ModelNotFoundError(f"Model not found: {model_id}")

                model_data = {
                    "name": metadata.name,
                    "version": metadata.version,
                    "stage": metadata.stage,
                    "created_at": metadata.created_at.isoformat(),
                    "model_type": metadata.model_type,
                    "task_type": metadata.task_type,
                    "model_size_bytes": metadata.model_size_bytes
                }

                # Add metrics
                if metrics:
                    model_data["metrics"] = {
                        k: v for k, v in metadata.metrics.items() if k in metrics
                    }
                else:
                    model_data["metrics"] = metadata.metrics.copy()

                comparison[model_id] = model_data

            return comparison

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelRegistryError(f"Failed to compare models: {str(e)}")

    def get_best_model(
        self,
        name: str,
        metric: str,
        stage: str | None = None,
        maximize: bool = True
    ) -> ModelVersion | None:
        """
        Get the best model by a specific metric.

        Args:
            name: Model name
            metric: Metric name to optimize
            stage: Stage filter
            maximize: Whether to maximize (True) or minimize (False) the metric

        Returns:
            Best model version or None if not found

        Raises:
            ModelRegistryError: If search fails
        """
        try:
            # Get all models with the specified name
            models = self.list_models(name_pattern=name, stage=stage, limit=1000)

            if not models:
                return None

            # Find best model by metric
            best_model = None
            best_value = None

            for metadata in models:
                if metric not in metadata.metrics:
                    continue

                value = metadata.metrics[metric]

                if best_value is None:
                    best_model = metadata
                    best_value = value
                elif (maximize and value > best_value) or (not maximize and value < best_value):
                    best_model = metadata
                    best_value = value

            if best_model:
                return self.get_model(best_model.model_id)

            return None

        except Exception as e:
            raise ModelRegistryError(f"Failed to find best model: {str(e)}")

    def load_model(self, model_id: str) -> Any:
        """
        Load a model object from storage.

        Args:
            model_id: Model identifier

        Returns:
            Loaded model object

        Raises:
            ModelNotFoundError: If model not found
            ModelStorageError: If loading fails
        """
        try:
            model_version = self.get_model(model_id)

            with open(model_version.model_path, 'rb') as f:
                model = pickle.load(f)

            # Verify model integrity
            expected_hash = model_version.metadata.model_hash
            actual_hash = self._calculate_model_hash(model_version.model_path)

            if expected_hash != actual_hash:
                self.logger.warning(
                    f"Model {model_id} hash mismatch. Expected: {expected_hash}, "
                    f"Actual: {actual_hash}. Model may be corrupted."
                )

            return model

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelStorageError(f"Failed to load model {model_id}: {str(e)}")

    def _generate_next_version(self, name: str) -> str:
        """Generate next semantic version for a model."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT version FROM models WHERE name = ? ORDER BY version DESC LIMIT 1",
                    (name,)
                )
                row = cursor.fetchone()

                if not row:
                    return "1.0.0"

                # Parse current version and increment
                current_version = row['version']
                try:
                    major, minor, patch = map(int, current_version.split('.'))
                    return f"{major}.{minor}.{patch + 1}"
                except ValueError:
                    # Fallback if version format is not semantic
                    return f"{current_version}_1"

        except Exception:
            # Fallback to timestamp-based version
            return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _store_model_artifacts(self, model_id: str, training_result: TrainingResult) -> Path:
        """Store model and associated artifacts."""
        model_dir = self.models_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Store model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(training_result.model, f)

        # Store additional artifacts
        artifacts_dir = model_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Store scaler if present
        if training_result.scaler:
            scaler_path = artifacts_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(training_result.scaler, f)

        # Store feature selector if present
        if training_result.feature_selector:
            selector_path = artifacts_dir / "feature_selector.pkl"
            with open(selector_path, 'wb') as f:
                pickle.dump(training_result.feature_selector, f)

        # Store label encoder if present
        if training_result.label_encoder:
            encoder_path = artifacts_dir / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(training_result.label_encoder, f)

        # Store training metadata
        metadata_path = artifacts_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_result.training_metadata, f, indent=2)

        return model_path

    def _cleanup_model_artifacts(self, model_id: str) -> None:
        """Clean up model artifacts from storage."""
        model_dir = self.models_path / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate SHA256 hash of model file."""
        hash_sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _store_metadata(self, metadata: ModelMetadata) -> None:
        """Store model metadata in database."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()

            # Check if model with same name and version already exists
            cursor.execute(
                "SELECT COUNT(*) as count FROM models WHERE name = ? AND version = ?",
                (metadata.name, metadata.version)
            )
            if cursor.fetchone()['count'] > 0:
                raise ModelVersionError(
                    f"Model {metadata.name} version {metadata.version} already exists"
                )

            # Insert main model record
            cursor.execute("""
                INSERT INTO models (
                    model_id, name, version, description, created_at, updated_at,
                    stage, model_type, task_type, model_size_bytes, model_hash,
                    parent_run_id, experiment_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.model_id, metadata.name, metadata.version, metadata.description,
                metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
                metadata.stage, metadata.model_type, metadata.task_type,
                metadata.model_size_bytes, metadata.model_hash,
                metadata.parent_run_id, metadata.experiment_name
            ))

            # Insert metrics
            for metric_name, metric_value in metadata.metrics.items():
                cursor.execute(
                    "INSERT INTO model_metrics (model_id, metric_name, metric_value) VALUES (?, ?, ?)",
                    (metadata.model_id, metric_name, metric_value)
                )

            # Insert hyperparameters
            for param_name, param_value in metadata.hyperparameters.items():
                cursor.execute(
                    "INSERT INTO model_hyperparameters (model_id, param_name, param_value) VALUES (?, ?, ?)",
                    (metadata.model_id, param_name, json.dumps(param_value))
                )

            # Insert tags
            for tag_key, tag_value in metadata.tags.items():
                cursor.execute(
                    "INSERT INTO model_tags (model_id, tag_key, tag_value) VALUES (?, ?, ?)",
                    (metadata.model_id, tag_key, tag_value)
                )

            # Insert feature names
            for idx, feature_name in enumerate(metadata.feature_names):
                cursor.execute(
                    "INSERT INTO model_features (model_id, feature_name, feature_index) VALUES (?, ?, ?)",
                    (metadata.model_id, feature_name, idx)
                )

            conn.commit()

    def _get_metadata(self, model_id: str) -> ModelMetadata | None:
        """Retrieve model metadata from database."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()

            # Get main model record
            cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()

            if not row:
                return None

            # Get metrics
            cursor.execute("SELECT metric_name, metric_value FROM model_metrics WHERE model_id = ?", (model_id,))
            metrics = {row['metric_name']: row['metric_value'] for row in cursor.fetchall()}

            # Get hyperparameters
            cursor.execute("SELECT param_name, param_value FROM model_hyperparameters WHERE model_id = ?", (model_id,))
            hyperparameters = {}
            for param_row in cursor.fetchall():
                try:
                    hyperparameters[param_row['param_name']] = json.loads(param_row['param_value'])
                except json.JSONDecodeError:
                    hyperparameters[param_row['param_name']] = param_row['param_value']

            # Get tags
            cursor.execute("SELECT tag_key, tag_value FROM model_tags WHERE model_id = ?", (model_id,))
            tags = {row['tag_key']: row['tag_value'] for row in cursor.fetchall()}

            # Get feature names
            cursor.execute(
                "SELECT feature_name FROM model_features WHERE model_id = ? ORDER BY feature_index",
                (model_id,)
            )
            feature_names = [row['feature_name'] for row in cursor.fetchall()]

            return ModelMetadata(
                model_id=row['model_id'],
                name=row['name'],
                version=row['version'],
                description=row['description'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                stage=row['stage'],
                model_type=row['model_type'],
                task_type=row['task_type'],
                model_size_bytes=row['model_size_bytes'],
                model_hash=row['model_hash'],
                parent_run_id=row['parent_run_id'],
                experiment_name=row['experiment_name'],
                metrics=metrics,
                hyperparameters=hyperparameters,
                tags=tags,
                feature_names=feature_names
            )

    def cleanup_old_models(
        self,
        keep_latest: int = 5,
        keep_production: bool = True,
        dry_run: bool = False
    ) -> list[str]:
        """
        Clean up old model versions.

        Args:
            keep_latest: Number of latest versions to keep per model name
            keep_production: Whether to keep production models
            dry_run: If True, return models that would be deleted without deleting

        Returns:
            List of model IDs that were (or would be) deleted

        Raises:
            ModelRegistryError: If cleanup fails
        """
        try:
            deleted_models = []

            # Get all model names
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT name FROM models")
                model_names = [row['name'] for row in cursor.fetchall()]

            for name in model_names:
                # Get all versions for this model name, ordered by creation time
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    query = "SELECT model_id, stage FROM models WHERE name = ? ORDER BY created_at DESC"
                    cursor.execute(query, (name,))
                    rows = cursor.fetchall()

                # Skip the latest N models
                models_to_check = rows[keep_latest:]

                for row in models_to_check:
                    model_id = row['model_id']
                    stage = row['stage']

                    # Skip production models if requested
                    if keep_production and stage == 'production':
                        continue

                    if dry_run:
                        deleted_models.append(model_id)
                    else:
                        try:
                            self.delete_model(model_id, force=True)
                            deleted_models.append(model_id)
                            self.logger.info(f"Cleaned up old model: {model_id}")
                        except Exception as e:
                            self.logger.error(f"Failed to delete model {model_id}: {str(e)}")

            return deleted_models

        except Exception as e:
            raise ModelRegistryError(f"Failed to cleanup old models: {str(e)}")

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                stats = {}

                # Total models
                cursor.execute("SELECT COUNT(*) as count FROM models")
                stats['total_models'] = cursor.fetchone()['count']

                # Models by stage
                cursor.execute("SELECT stage, COUNT(*) as count FROM models GROUP BY stage")
                stats['models_by_stage'] = {row['stage']: row['count'] for row in cursor.fetchall()}

                # Models by type
                cursor.execute("SELECT model_type, COUNT(*) as count FROM models GROUP BY model_type")
                stats['models_by_type'] = {row['model_type']: row['count'] for row in cursor.fetchall()}

                # Total storage size
                cursor.execute("SELECT SUM(model_size_bytes) as total_size FROM models")
                total_size = cursor.fetchone()['total_size'] or 0
                stats['total_storage_bytes'] = total_size
                stats['total_storage_mb'] = round(total_size / (1024 * 1024), 2)

                # Registry disk usage
                registry_size = sum(
                    f.stat().st_size for f in self.models_path.rglob('*') if f.is_file()
                )
                stats['actual_storage_bytes'] = registry_size
                stats['actual_storage_mb'] = round(registry_size / (1024 * 1024), 2)

                return stats

        except Exception as e:
            raise ModelRegistryError(f"Failed to get registry stats: {str(e)}")

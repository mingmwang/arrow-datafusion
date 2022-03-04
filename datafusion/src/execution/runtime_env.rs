// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Execution runtime environment that tracks memory, disk and various configurations
//! that are used during physical plan execution.

use crate::{
    error::Result,
    execution::{
        disk_manager::{DiskManager, DiskManagerConfig},
        memory_manager::{MemoryConsumerId, MemoryManager, MemoryManagerConfig},
    },
};
use std::fmt;
use std::fmt::{Debug, Formatter};

use crate::datasource::object_store::{ObjectStore, ObjectStoreRegistry};
use crate::execution::context::{
    SessionContextRegistry, TaskContext, TaskContextRegistry, BATCH_SIZE,
    PARQUET_PRUNING, REPARTITION_AGGREGATIONS, REPARTITION_JOINS, REPARTITION_WINDOWS,
    TARGET_PARTITIONS,
};
use crate::prelude::{SessionConfig, SessionContext};
use datafusion_common::DataFusionError;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

/// Global singleton RuntimeEnv
pub static RUNTIME_ENV: OnceCell<Arc<RuntimeEnv>> = OnceCell::new();

/// Execution runtime environment. This structure is a singleton for each Scheduler/Executor instance.
pub struct RuntimeEnv {
    /// Executor Id
    pub executor_id: Option<String>,
    /// Local Env
    pub is_local: bool,
    /// Runtime memory management
    pub memory_manager: Arc<MemoryManager>,
    /// Manage temporary files during query execution
    pub disk_manager: Arc<DiskManager>,
    /// Object Store that are registered within the Scheduler's or Executors' Runtime
    pub object_store_registry: Arc<ObjectStoreRegistry>,
    /// DataFusion task contexts that are registered within the Executors' Runtime
    pub task_context_registry: Option<Arc<TaskContextRegistry>>,
    /// DataFusion session contexts that are registered within the Scheduler's Runtime
    pub session_context_registry: Option<Arc<SessionContextRegistry>>,
}

impl RuntimeEnv {
    /// Create an executor env based on configuration
    pub fn new_executor_env(config: RuntimeConfig, executor_id: String) -> Result<Self> {
        let RuntimeConfig {
            memory_manager,
            disk_manager,
        } = config;
        Ok(Self {
            executor_id: Some(executor_id),
            is_local: false,
            memory_manager: MemoryManager::new(memory_manager),
            disk_manager: DiskManager::try_new(disk_manager)?,
            object_store_registry: Arc::new(ObjectStoreRegistry::new()),
            task_context_registry: Some(Arc::new(TaskContextRegistry::new())),
            session_context_registry: None,
        })
    }

    /// Create a scheduler env based on configuration
    pub fn new_scheduler_env(config: RuntimeConfig) -> Result<Self> {
        let RuntimeConfig {
            memory_manager,
            disk_manager,
        } = config;
        Ok(Self {
            executor_id: None,
            is_local: false,
            memory_manager: MemoryManager::new(memory_manager),
            disk_manager: DiskManager::try_new(disk_manager)?,
            object_store_registry: Arc::new(ObjectStoreRegistry::new()),
            task_context_registry: None,
            session_context_registry: Some(Arc::new(SessionContextRegistry::new())),
        })
    }

    /// Create a local env based on configuration
    pub fn new_local_env(config: RuntimeConfig) -> Result<Self> {
        let RuntimeConfig {
            memory_manager,
            disk_manager,
        } = config;
        Ok(Self {
            executor_id: None,
            is_local: true,
            memory_manager: MemoryManager::new(memory_manager),
            disk_manager: DiskManager::try_new(disk_manager)?,
            object_store_registry: Arc::new(ObjectStoreRegistry::new()),
            task_context_registry: None,
            session_context_registry: Some(Arc::new(SessionContextRegistry::new())),
        })
    }

    /// Return the global singleton RuntimeEnv
    pub fn global() -> &'static Arc<RuntimeEnv> {
        RUNTIME_ENV.get_or_init(|| {
            Arc::new(RuntimeEnv::new_local_env(RuntimeConfig::new()).unwrap())
        })
    }

    /// Is Scheduler RuntimeEnv
    pub fn is_scheduler(&self) -> bool {
        self.is_local || self.executor_id.is_none()
    }

    /// Is Local RuntimeEnv
    pub fn is_local(&self) -> bool {
        self.is_local
    }

    /// Register the consumer to get it tracked
    pub fn register_requester(&self, id: &MemoryConsumerId) {
        self.memory_manager.register_requester(id);
    }

    /// Drop the consumer from get tracked, reclaim memory
    pub fn drop_consumer(&self, id: &MemoryConsumerId, mem_used: usize) {
        self.memory_manager.drop_consumer(id, mem_used)
    }

    /// Grow tracker memory of `delta`
    pub fn grow_tracker_usage(&self, delta: usize) {
        self.memory_manager.grow_tracker_usage(delta)
    }

    /// Shrink tracker memory of `delta`
    pub fn shrink_tracker_usage(&self, delta: usize) {
        self.memory_manager.shrink_tracker_usage(delta)
    }

    /// Registers a object store with scheme using a custom `ObjectStore` so that
    /// an external file system or object storage system could be used against this context.
    ///
    /// Returns the `ObjectStore` previously registered for this scheme, if any
    pub fn register_object_store(
        &self,
        scheme: impl Into<String>,
        object_store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>> {
        let scheme = scheme.into();
        self.object_store_registry
            .register_store(scheme, object_store)
    }

    /// Retrieves a `ObjectStore` instance by scheme
    pub fn object_store<'a>(
        &self,
        uri: &'a str,
    ) -> Result<(Arc<dyn ObjectStore>, &'a str)> {
        self.object_store_registry
            .get_by_uri(uri)
            .map_err(DataFusionError::from)
    }

    /// Retrieves a copied version of `SessionConfig` by session_id
    pub fn lookup_session_config(&self, session_id: &str) -> SessionConfig {
        if self.is_local() {
            // It is possible that in a local env such as in unit tests there is no
            // SessionContext created, in this case we have to return a default SessionConfig.
            let session_conf = self
                .lookup_config(session_id)
                .map_or(SessionConfig::new(), |c| c.lock().clone());
            session_conf
        } else if self.is_scheduler() {
            let session_conf = self
                .lookup_session(session_id)
                .expect("SessionContext doesn't exist")
                .copied_config();
            session_conf
        } else {
            self.config_from_task_context(session_id)
        }
    }

    /// Registers a `SessionContext` with session_id.
    /// TODO panic if the SessionContext with session_id already exists
    pub fn register_session(
        &self,
        session_id: String,
        session_context: Arc<SessionContext>,
    ) -> Option<Arc<SessionContext>> {
        self.session_context_registry.as_ref().expect(
            "SessionContextRegistry is not initialized, should not be called in an Executor.")
            .register_session(session_id, session_context)
    }

    /// Retrieves a `SessionContext` instance by session_id
    pub fn lookup_session(&self, session_id: &str) -> Option<Arc<SessionContext>> {
        self.session_context_registry.as_ref().expect(
            "SessionContextRegistry is not initialized, should not be called in an Executor.")
            .lookup_session(session_id)
    }

    /// Remove a a `SessionContext` instance by session_id
    pub fn unregister_session(&self, session_id: &str) -> Option<Arc<SessionContext>> {
        self.session_context_registry.as_ref().expect(
            "SessionContextRegistry is not initialized, should not be called in an Executor.")
            .unregister_session(session_id)
    }

    /// Registers a `SessionConfig` with session_id.
    /// TODO panic if the SessionConfig with session_id already exists
    pub fn register_config(
        &self,
        session_id: String,
        session_config: Arc<Mutex<SessionConfig>>,
    ) -> Option<Arc<Mutex<SessionConfig>>> {
        self.session_context_registry.as_ref().expect(
            "SessionContextRegistry is not initialized, should not be called in an Executor.")
            .register_config(session_id, session_config)
    }

    /// Retrieves a `SessionConfig` instance by session_id
    pub fn lookup_config(&self, session_id: &str) -> Option<Arc<Mutex<SessionConfig>>> {
        self.session_context_registry.as_ref().expect(
            "SessionContextRegistry is not initialized, should not be called in an Executor.")
            .lookup_config(session_id)
    }

    /// Remove a a `SessionConfig` instance by session_id
    pub fn unregister_config(
        &self,
        session_id: &str,
    ) -> Option<Arc<Mutex<SessionConfig>>> {
        self.session_context_registry.as_ref().expect(
            "SessionContextRegistry is not initialized, should not be called in an Executor.")
            .unregister_config(session_id)
    }

    /// Registers a `TaskContext` with task_id.
    /// TODO panic if the TaskContext with session_id already exists
    pub fn register_task(
        &self,
        task_id: String,
        task_context: TaskContext,
    ) -> Option<Arc<TaskContext>> {
        self.task_context_registry.as_ref().expect(
            "TaskContextRegistry is not initialized, should not be called in an Scheduler.")
            .register_task(task_id, task_context)
    }

    /// Retrieves a `TaskContext` instance by task_id
    pub fn lookup_task(&self, task_id: &str) -> Option<Arc<TaskContext>> {
        self.task_context_registry.as_ref().expect(
            "TaskContextRegistry is not initialized, should not be called in an Scheduler.")
            .lookup_task(task_id)
    }

    /// Remove a a `TaskContext` instance by task_id
    pub fn unregister_task(&self, task_id: &str) -> Option<Arc<TaskContext>> {
        self.task_context_registry.as_ref().expect(
            "SessionContextRegistry is not initialized, should not be called in an Scheduler.")
            .unregister_task(task_id)
    }

    fn config_from_task_context(&self, session_id: &str) -> SessionConfig {
        let task_ctx = self.task_context_registry.as_ref().expect(
            "TaskContextRegistry is not initialized, should not be called in an Scheduler.")
            .lookup_task_for_session(session_id).expect("No task exists for Session.");
        let props = &task_ctx.task_settings;
        let session_config = SessionConfig::new();
        session_config
            .with_batch_size(props.get(BATCH_SIZE).unwrap().parse().unwrap())
            .with_target_partitions(
                props.get(TARGET_PARTITIONS).unwrap().parse().unwrap(),
            )
            .with_repartition_joins(
                props.get(REPARTITION_JOINS).unwrap().parse().unwrap(),
            )
            .with_repartition_aggregations(
                props
                    .get(REPARTITION_AGGREGATIONS)
                    .unwrap()
                    .parse()
                    .unwrap(),
            )
            .with_repartition_windows(
                props.get(REPARTITION_WINDOWS).unwrap().parse().unwrap(),
            )
            .with_parquet_pruning(props.get(PARQUET_PRUNING).unwrap().parse().unwrap())
    }
}

impl Debug for RuntimeEnv {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("RuntimeEnv")
            .field("executor_id", &self.executor_id)
            .field("is_local", &self.is_local())
            .field("is_scheduler", &self.is_scheduler())
            .finish()
    }
}

#[derive(Clone)]
/// Execution runtime configuration
pub struct RuntimeConfig {
    /// DiskManager to manage temporary disk file usage
    pub disk_manager: DiskManagerConfig,
    /// MemoryManager to limit access to memory
    pub memory_manager: MemoryManagerConfig,
}

impl RuntimeConfig {
    /// New with default values
    pub fn new() -> Self {
        Default::default()
    }

    /// Customize disk manager
    pub fn with_disk_manager(mut self, disk_manager: DiskManagerConfig) -> Self {
        self.disk_manager = disk_manager;
        self
    }

    /// Customize memory manager
    pub fn with_memory_manager(mut self, memory_manager: MemoryManagerConfig) -> Self {
        self.memory_manager = memory_manager;
        self
    }

    /// Specify the total memory to use while running the DataFusion
    /// plan to `max_memory * memory_fraction` in bytes.
    ///
    /// Note DataFusion does not yet respect this limit in all cases.
    pub fn with_memory_limit(self, max_memory: usize, memory_fraction: f64) -> Self {
        let conf = self.with_memory_manager(
            MemoryManagerConfig::try_new_limit(max_memory, memory_fraction).unwrap(),
        );
        conf
    }

    /// Use the specified path to create any needed temporary files
    pub fn with_temp_file_path(self, path: impl Into<PathBuf>) -> Self {
        let conf =
            self.with_disk_manager(DiskManagerConfig::new_specified(vec![path.into()]));
        conf
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            disk_manager: DiskManagerConfig::default(),
            memory_manager: MemoryManagerConfig::default(),
        }
    }
}

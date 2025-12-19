/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use std::{sync::Arc, thread, time::Duration};

use dashmap::DashMap;
use rand::Rng;
use tracing::{debug, info};

use super::{get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy};
use crate::{core::Worker, metrics::RouterMetrics, tree::Tree};

use serde::{Deserialize, Serialize};  
use tokio::task::JoinHandle;  
use crate::tokenizer::traits::Tokenizer;

use std::sync::RwLock;
use std::collections::HashMap;  

// 添加响应数据结构  
#[derive(Debug, Deserialize, Serialize)]  
struct RadixTreeResponse {  
    ops_id: u64,  
    ops_id_finished: u64,  
    instance_id: String,  
    node_ip: String,  
    server_port: u16,  
    tree: RadixTreeNode,  
}  
  
#[derive(Debug, Deserialize, Serialize)]  
struct RadixTreeNode {  
    key: Vec<u32>,  
    value: Vec<u32>,  
    children: Vec<RadixTreeNode>,  
} 

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    eviction_handle: Option<thread::JoinHandle<()>>,
    sync_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
    sync_tasks: Arc<RwLock<HashMap<String, JoinHandle<()>>>>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        // Start background eviction thread if configured
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            let interval = config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));

                // Evict for all model trees
                for tree_ref in trees_clone.iter() {
                    let model_id = tree_ref.key();
                    let tree = tree_ref.value();
                    tree.evict_tenant_by_size(max_tree_size);
                    debug!(
                        "Cache eviction completed for model {}, max_size: {}",
                        model_id, max_tree_size
                    );
                }
            }))
        } else {
            None
        };

        Self {
            config,
            trees,
            eviction_handle,
            sync_handle: Arc::new(RwLock::new(None)),
            sync_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 获取配置的不可变引用  
    pub fn config(&self) -> &CacheAwareConfig {  
        &self.config  
    } 

    /// 检查是否启用了缓存同步功能  
    pub fn is_cache_sync_enabled(&self) -> bool {  
        self.config.enable_cache_sync  
    }  

    /// 获取同步间隔(秒)  
    pub fn sync_interval_secs(&self) -> u64 {  
        self.config.sync_interval_secs  
    }

    /// 获取缓存阈值  
    pub fn cache_threshold(&self) -> f32 {  
        self.config.cache_threshold  
    }

    /// 获取最大树大小  
    pub fn max_tree_size(&self) -> usize {  
        self.config.max_tree_size  
    } 

    /// 启动缓存同步任务 (仅在 enable_cache_sync 为 true 时调用) 
    pub fn start_cache_sync(  
        &self,  
        prefill_worker_url: String,  
        tokenizer: Arc<dyn Tokenizer>,  
    ) {

        if !self.config.enable_cache_sync {  
            debug!("Cache sync is disabled, skipping sync task initialization");  
            return;  
        }    
        
        // 检查是否已有同步任务  
        {  
            let mut tasks = self.sync_tasks.write().unwrap();  
            if tasks.contains_key(&prefill_worker_url) {  
                info!("Cache sync already running for worker: {}", prefill_worker_url);  
                return;  
            }  
        } 

        let trees = Arc::clone(&self.trees);  
        let sync_interval_secs = self.config.sync_interval_secs;  
        let worker_url_for_log = prefill_worker_url.clone();
        let sync_tasks = Arc::clone(&self.sync_tasks); 

        // 在移动到闭包前克隆一个
        let worker_url_for_closure = prefill_worker_url.clone();

        let handle = tokio::spawn(async move {    
            let mut interval = tokio::time::interval(    
                tokio::time::Duration::from_secs(sync_interval_secs)    
            );    
                
            loop {    
                interval.tick().await; 
                // 根据URL判断连接模式并选择获取方式  
                let tree_result = if worker_url_for_closure.starts_with("grpc://") || worker_url_for_closure.starts_with("grpcs://") {  
                    fetch_cache_tree_from_grpc_worker(&worker_url_for_closure).await  
                } else {  
                    fetch_cache_tree_from_http_worker(&worker_url_for_closure).await  
                };  
                
                match tree_result {    
                    Ok(tree_response) => {    
                        tracing::info!(    
                            "Fetched cache tree from {} (ops_id: {}, instance_id: {})",    
                            worker_url_for_closure,    
                            tree_response.ops_id,    
                            tree_response.instance_id,  
                        );    
                            
                        match detokenize_tree(&tree_response, &tokenizer) {    
                            Ok(string_tree) => {    
                                replace_tree(&trees, &tree_response, string_tree);    
                            }    
                            Err(e) => {    
                                tracing::warn!(    
                                    "Failed to detokenize tree from {}: {}",    
                                    worker_url_for_closure,    
                                    e    
                                );    
                            }    
                        }    
                    }    
                    Err(e) => {    
                        tracing::warn!(    
                            "Failed to fetch cache tree from {}: {}",     
                            worker_url_for_closure,     
                            e    
                        );    
                    }    
                }    
            }    
        });  
          
        sync_tasks.write().unwrap().insert(prefill_worker_url, handle);

        tracing::info!(  
            "Cache sync enabled for worker {} with interval {} seconds",  
            worker_url_for_log,  
            sync_interval_secs  
        );  
    }

    /// 停止指定worker的缓存同步  
    pub fn stop_cache_sync(&self, worker_url: &str) {  
        if let Some(handle) = self.sync_tasks.write().unwrap().remove(worker_url) {  
            handle.abort();  
            info!("Cache sync stopped for worker: {}", worker_url);  
        }  
    }  
      
    /// 检查worker是否有活跃的同步任务  
    pub fn has_active_sync_task(&self, worker_url: &str) -> bool {  
        self.sync_tasks.read().unwrap().contains_key(worker_url)  
    }

    /// Initialize the tree with worker URLs (used only during initial setup)
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        let mut model_workers: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            // Use "default" for unknown/empty model_ids for backward compatibility
            let model_id = worker.model_id();
            let tree_key = if model_id.is_empty() || model_id == "unknown" {
                "default"
            } else {
                model_id
            };
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize tree for each model
        for (tree_key, model_workers) in model_workers {
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()));
            for worker in model_workers {
                tree.insert("", worker.url());
            }
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        // For backward compatibility: if model_id is "unknown" or empty,
        // use a default tree. This preserves existing behavior for single-model routers.
        let model_id = worker.model_id();
        let tree_key = if model_id.is_empty() || model_id == "unknown" {
            "default"
        } else {
            model_id
        };
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", url);
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        // Use same logic as add_worker for consistency
        let model_id = worker.model_id();
        let tree_key = if model_id.is_empty() || model_id == "unknown" {
            "default"
        } else {
            model_id
        };
        if let Some(tree) = self.trees.get(tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Cache eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
    }
}

// HTTP模式缓存获取函数  
async fn fetch_cache_tree_from_http_worker(    
    worker_url: &str    
) -> Result<RadixTreeResponse, Box<dyn std::error::Error>> {    
    let client = reqwest::Client::new();    
    let response = client    
        .get(format!("{}/v1/radixtree/full", worker_url))    
        .send()    
        .await?;    
        
    let tree_response: RadixTreeResponse = response.json().await?;    
    Ok(tree_response)    
}    
  
// gRPC模式缓存获取函数（占位符，需要根据实际gRPC接口实现）  
async fn fetch_cache_tree_from_grpc_worker(    
    worker_url: &str    
) -> Result<RadixTreeResponse, Box<dyn std::error::Error>> {    
    // TODO: 实现gRPC模式的缓存树获取 
    Err("gRPC mode cache sync not yet implemented".into())  
}

// mock 
// 临时硬编码响应用于测试
// TODO: 等 /v1/radixtree/full 接口在 prefill worker 上实现后删除此 mock 
async fn mock_fetch_cache_tree_from_worker(    
    worker_url: &str,  
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<RadixTreeResponse, Box<dyn std::error::Error>> {    
     
    // 预设测试提示词  
    let test_prompt = "Hello, how are you?";  
      
    // 使用 tokenizer 进行 token 化  
    let encoding = tokenizer.encode(test_prompt)?;  
    let token_ids: Vec<u32> = encoding.token_ids().to_vec();  
      
    info!(  
        "Mock: tokenized '{}' into {} tokens: {:?}",   
        test_prompt,   
        token_ids.len(),  
        &token_ids[..token_ids.len().min(10)]  // 只打印前10个  
    );  
      
    let mock_response = RadixTreeResponse {  
        ops_id: 1,  
        ops_id_finished: 1,  
        instance_id: "worker-8000".to_string(),  
        node_ip: "127.0.0.1".to_string(),  
        server_port: 8000,  
        tree: RadixTreeNode {  
            key: token_ids,  // 使用真实的 token IDs  
            value: vec![],  
            children: vec![],  
        },  
    };  
      
    info!("使用 mock 缓存树响应,worker: {}", worker_url);  
    return Ok(mock_response);     
}
  
fn detokenize_tree(  
    tree_response: &RadixTreeResponse,  
    tokenizer: &Arc<dyn Tokenizer>,  
) -> Result<Tree, Box<dyn std::error::Error>> {  
    let new_tree = Tree::new();  
    let tenant = format!("{}:{}", tree_response.node_ip, tree_response.server_port);  
      
    fn process_node(  
        node: &RadixTreeNode,  
        tree: &Tree,  
        tokenizer: &Arc<dyn Tokenizer>,  
        tenant: &str,  
        accumulated_tokens: &mut Vec<u32>,  
    ) -> Result<(), Box<dyn std::error::Error>> {  
        accumulated_tokens.extend_from_slice(&node.key);  
          
        if !accumulated_tokens.is_empty() {  
            let text = tokenizer.decode(accumulated_tokens, false)?;

            // 添加验证日志: 打印 token IDs 和反 token 化后的文本  
            info!(  
                "Detokenized {} tokens {:?} -> text: '{}'",  
                accumulated_tokens.len(),  
                &accumulated_tokens[..accumulated_tokens.len().min(20)],  // 最多打印前20个  
                text  
            );

            tree.insert(&text, tenant);  
        }  
          
        for child in &node.children {  
            let mut child_tokens = accumulated_tokens.clone();  
            process_node(child, tree, tokenizer, tenant, &mut child_tokens)?;  
        }  
          
        Ok(())  
    }  
      
    let mut initial_tokens = Vec::new();  
    process_node(&tree_response.tree, &new_tree, tokenizer, &tenant, &mut initial_tokens)?;  
    Ok(new_tree)  
}  
  
fn replace_tree(  
    trees: &Arc<DashMap<String, Arc<Tree>>>,  
    tree_response: &RadixTreeResponse,  
    new_tree: Tree,  
) {  
    let model_id = "default";  
    trees.insert(model_id.to_string(), Arc::new(new_tree));  
      
    tracing::info!(  
        "Replaced cache tree for instance {} ({}:{}) in model {}, ops_id: {}",  
        tree_response.instance_id,  
        tree_response.node_ip,  
        tree_response.server_port,  
        model_id,  
        tree_response.ops_id  
    );  
}  
  
impl Drop for CacheAwarePolicy {  
    fn drop(&mut self) {  
        if let Some(handle) = self.eviction_handle.take() {  
            drop(handle);  
        }  

        // 停止所有同步任务  
        for (worker_url, handle) in self.sync_tasks.write().unwrap().drain() {  
            handle.abort();  
            info!("Cache sync stopped for worker: {}", worker_url);  
        }
          
        if let Ok(mut guard) = self.sync_handle.write() {  
            if let Some(handle) = guard.take() {  
                handle.abort();  
            }  
        }  
    }  
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let first_model = workers[healthy_indices[0]].model_id();
        let model_id = if first_model.is_empty() || first_model == "unknown" {
            "default"
        } else {
            first_model
        };

        // Get current load statistics
        let loads: Vec<usize> = workers.iter().map(|w| w.load()).collect();
        let max_load = *loads.iter().max().unwrap_or(&0);
        let min_load = *loads.iter().min().unwrap_or(&0);

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            // Log load balancing trigger
            let worker_loads: Vec<(String, usize)> = workers
                .iter()
                .map(|w| (w.url().to_string(), w.load()))
                .collect();

            debug!(
                "Load balancing triggered | max: {} | min: {} | workers: {:?}",
                max_load, min_load, worker_loads
            );

            RouterMetrics::record_load_balancing_event();
            RouterMetrics::set_load_range(max_load, min_load);

            // Use shortest queue when imbalanced
            let min_load_idx = healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()?;

            // Even in imbalanced mode, update the tree to maintain cache state
            if let Some(text) = request_text {
                // Get the tree reference without locking the entire HashMap
                // DashMap only locks the specific shard containing this key
                let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

                if let Some(tree) = tree {
                    // Now we can work with the tree without holding the HashMap lock
                    tree.insert(text, workers[min_load_idx].url());
                } else {
                    debug!(
                        "Warning: No tree found for model '{}', skipping cache update",
                        model_id
                    );
                }
            }

            // Increment processed counter
            workers[min_load_idx].increment_processed();
            RouterMetrics::record_processed_request(workers[min_load_idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[min_load_idx].url());

            return Some(min_load_idx);
        }

        // Use cache-aware routing when balanced
        let text = request_text.unwrap_or("");

        // Get the tree reference without locking the entire HashMap
        // DashMap only locks the specific shard containing this key
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            // Now we work with the tree without holding the HashMap lock
            let (matched_text, matched_worker) = tree.prefix_match(text);
            let match_rate = if text.is_empty() {
                0.0
            } else {
                matched_text.chars().count() as f32 / text.chars().count() as f32
            };

            let selected_url = if match_rate > self.config.cache_threshold {
                RouterMetrics::record_cache_hit();
                matched_worker.to_string()
            } else {
                RouterMetrics::record_cache_miss();
                tree.get_smallest_tenant()
            };

            // Find the index of the selected worker
            if let Some(selected_idx) = workers.iter().position(|w| w.url() == selected_url) {
                // Only proceed if the worker is healthy
                if workers[selected_idx].is_healthy() {
                    // Update the tree with this request
                    tree.insert(text, &selected_url);

                    // Increment processed counter
                    workers[selected_idx].increment_processed();
                    RouterMetrics::record_processed_request(&selected_url);

                    return Some(selected_idx);
                }
            } else {
                // Selected worker no longer exists, remove it from tree
                tree.remove_tenant(&selected_url);
                debug!("Removed stale worker {} from cache tree", selected_url);
            }

            // Fallback to first healthy worker
            healthy_indices.first().copied()
        } else {
            // No tree for this model, log warning and use random selection
            debug!(
                "Warning: No tree found for model '{}', using random worker selection",
                model_id
            );
            // Return a random healthy worker
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn select_worker_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        // DEPRECATED: This method is no longer used when separate policies are configured.
        // The PD router now uses separate policies for prefill and decode selection.
        // This implementation remains for backward compatibility when a single policy is used.

        // In PD mode with single policy:
        // - Prefill: Use cache-aware routing for better cache utilization
        // - Decode: Use least-load routing for better load distribution

        // Select prefill worker using cache-aware logic
        let prefill_idx = self.select_worker(prefill_workers, request_text)?;

        // Select decode worker using least-load logic
        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let decode_idx = healthy_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy.select_worker(&workers, Some("hello world")).unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy.select_worker(&workers, Some("hello world")).unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy.select_worker(&workers, Some("hello")).unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
        });

        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, Some("test")).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(&workers, Some("test1"));
        policy.select_worker(&workers, Some("test2"));

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy.select_worker(&workers, Some("test1")).unwrap();
        assert_eq!(idx, 1);
    }
}

/*  
    Token Precise Match Load Balancing Router  
      
    This policy is designed specifically for PD (Prefill-Decode) separation in gRPC mode.  
    It combines load balancing with token-based precise matching via Nexus API.  
      
    Strategy Details:  
    1. Load Balancing (reused from cache_aware)  
       - Uses shortest queue when system is imbalanced  
       - Imbalance conditions: (max_load - min_load) > balance_abs_threshold   
         AND max_load > min_load * balance_rel_threshold  
      
    2. Token Precise Matching  
       - Calls /v1/Nexus/get_best_instance API with token ID list  
       - Returns optimal worker ID list for token matching  
*/  
  
use std::sync::Arc;  
use async_trait::async_trait;  
use serde::{Deserialize, Serialize};  
use tracing::{debug, error, info, warn};  
use reqwest::Client;  
use tokio::runtime::Runtime; 
  
use super::{get_healthy_worker_indices, LoadBalancingPolicy};  
use crate::core::Worker;  
  
#[derive(Debug, Clone)]  
pub struct TokenPreciseMatchConfig {  
    pub balance_abs_threshold: usize,  
    pub balance_rel_threshold: f32,  
    pub nexus_endpoint: String,  
    pub request_timeout_secs: u64,  
}  
  
impl Default for TokenPreciseMatchConfig {  
    fn default() -> Self {  
        Self {  
            balance_abs_threshold: 64,  
            balance_rel_threshold: 1.5,  
            nexus_endpoint: "http://localhost:8080".to_string(),  
            request_timeout_secs: 5,  
        }  
    }  
}  
  
#[derive(Debug, Clone)]  
pub struct TokenPreciseMatchPolicy {  
    config: TokenPreciseMatchConfig,  
    http_client: Client,  
}  
  
#[derive(Serialize)]  
struct NexusRequest {  
    token_ids: Vec<u32>,  
}  
  
#[derive(Deserialize)]  
struct NexusResponse {  
    worker_ids: Vec<String>,  
}  
  
impl TokenPreciseMatchPolicy {  
    pub fn new() -> Self {  
        Self::with_config(TokenPreciseMatchConfig::default())  
    }  
  
    pub fn with_config(config: TokenPreciseMatchConfig) -> Self {  
        let http_client = Client::builder()  
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))  
            .build()  
            .expect("Failed to create HTTP client");  
        
        // Create a tokio runtime for blocking async calls  
        let runtime = Arc::new(  
            Runtime::new()  
                .expect("Failed to create tokio runtime")  
        );
  
        Self {  
            config,  
            http_client,
            runtime,
        }  
    }  
  
    fn get_best_instance_from_nexus(&self, token_ids: &[u32]) -> Result<Vec<String>, String> {  
        let request = NexusRequest {  
            token_ids: token_ids.to_vec(),  
        };  
  
        let url = format!("{}/v1/Nexus/get_best_instance", self.config.nexus_endpoint);  
          
        debug!("Calling Nexus API at {} with {} tokens", url, token_ids.len());  
  
        // Use block_on to call async function from sync context
        let response = self.runtime.block_on(async {  
            self.http_client  
            .post(&url)  
            .json(&request)  
            .send()  
            .await  
        }).map_err(|e| format!("Failed to call Nexus API: {}", e))?;   
  
        if !response.status().is_success() {  
            return Err(format!("Nexus API returned status: {}", response.status()));  
        }  
  
        let nexus_response: NexusResponse = self.runtime.block_on(async {
            response.json().await 
        }).map_err(|e| format!("Failed to parse Nexus response: {}", e))?;  
  
        info!("Nexus returned {} worker candidates", nexus_response.worker_ids.len());  
        Ok(nexus_response.worker_ids)  
    }  

    /// Find worker index by worker ID from the available workers  
    fn find_worker_by_id(&self, workers: &[Arc<dyn Worker>], worker_id: &str) -> Option<usize> {  
        workers.iter().position(|w| {  
            w.metadata().id == worker_id ||   
            w.metadata().url.contains(worker_id) ||  
            w.metadata().url.ends_with(worker_id)  
        })  
    }
}  
  
#[async_trait]  
impl LoadBalancingPolicy for TokenPreciseMatchPolicy {  
    fn select_worker(  
        &self,  
        workers: &[Arc<dyn Worker>],  
        request_text: Option<&str>,  
    ) -> Option<usize> {  
        // This policy should only be used in PD gRPC mode  
        // For regular mode, fall back to load balancing  
        self.select_worker_with_tokens(workers, request_text, &[])  
    }  
    
    fn select_worker_with_tokens(  
        &self,  
        workers: &[Arc<dyn Worker>],  
        _request_text: Option<&str>,  
        token_ids: &[u32],  
    ) -> Option<usize> {  
        let healthy_indices = get_healthy_worker_indices(workers);  
  
        if healthy_indices.is_empty() {  
            return None;  
        }  
  
        // Get current load statistics  
        let loads: Vec<usize> = workers.iter().map(|w| w.load()).collect();  
        let max_load = *loads.iter().max().unwrap_or(&0);  
        let min_load = *loads.iter().min().unwrap_or(&0);  
  
        // Check if load is imbalanced (reuse cache_aware logic)  
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold  
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);  
  
        if is_imbalanced {  
            debug!(  
                "Load balancing triggered | max: {} | min: {}",  
                max_load, min_load  
            );  
  
            // Use shortest queue when imbalanced  
            let min_load_idx = healthy_indices  
                .iter()  
                .min_by_key(|&&idx| workers[idx].load())  
                .copied()?;  
  
            workers[min_load_idx].increment_processed();  
            return Some(min_load_idx);  
        }  
  
        // Use token precise matching when balanced  
        if !token_ids.is_empty() {  
            debug!("Using token-based matching with {} tokens", token_ids.len());  
              
            // Call Nexus API to get best worker  
            match self.get_best_instance_from_nexus(token_ids) {  
                Ok(worker_ids) => {  
                    // Try to find the first available worker from Nexus recommendations  
                    for worker_id in worker_ids {  
                        if let Some(idx) = self.find_worker_by_id(workers, &worker_id) {  
                            if healthy_indices.contains(&idx) {  
                                debug!("Selected worker {} based on Nexus recommendation", worker_id);  
                                workers[idx].increment_processed();  
                                return Some(idx);  
                            }  
                        }  
                    }  
                      
                    // If no recommended worker is available, fall back to load balancing  
                    warn!("No Nexus-recommended workers available, falling back to load balancing");  
                    let min_load_idx = healthy_indices  
                        .iter()  
                        .min_by_key(|&&idx| workers[idx].load())  
                        .copied()?;  
                    workers[min_load_idx].increment_processed();  
                    return Some(min_load_idx);  
                }  
                Err(e) => {  
                    error!("Failed to get worker from Nexus: {}, falling back to load balancing", e);  
                    // Fall back to load balancing on error  
                    let min_load_idx = healthy_indices  
                        .iter()  
                        .min_by_key(|&&idx| workers[idx].load())  
                        .copied()?;  
                    workers[min_load_idx].increment_processed();  
                    return Some(min_load_idx);  
                }  
            }    
        } else {  
            // No tokens available, use load balancing  
            debug!("No tokens available, using load balancing");  
            let min_load_idx = healthy_indices  
                .iter()  
                .min_by_key(|&&idx| workers[idx].load())  
                .copied()?;  
            workers[min_load_idx].increment_processed();  
            return Some(min_load_idx);
        }  
    }

    fn select_worker_pair(  
        &self,  
        prefill_workers: &[Arc<dyn Worker>],  
        decode_workers: &[Arc<dyn Worker>],  
        request_text: Option<&str>,  
    ) -> Option<(usize, usize)> {
        self.select_worker_pair_with_tokens(prefill_workers, decode_workers, request_text, &[])  
    }  

    fn select_worker_pair_with_tokens(  
        &self,  
        prefill_workers: &[Arc<dyn Worker>],  
        decode_workers: &[Arc<dyn Worker>],  
        _request_text: Option<&str>,  
        token_ids: &[u32],  
    ) -> Option<(usize, usize)> {  
        // For prefill workers, use token precise matching  
        let prefill_idx = self.select_worker_with_tokens(prefill_workers, None, token_ids)?;  
  
        // For decode workers, always use load balancing (no tokens)  
        let decode_idx = self.select_worker_with_tokens(decode_workers, None, &[])?;  
  
        Some((prefill_idx, decode_idx))  
    }
  
    fn name(&self) -> &'static str {  
        "token_precise_match"  
    }  
  
    fn needs_request_text(&self) -> bool {  
        true // This policy needs request text for tokenization  
    }  
  
    fn as_any(&self) -> &dyn std::any::Any {  
        self  
    }  
}  
  
impl Default for TokenPreciseMatchPolicy {  
    fn default() -> Self {  
        Self::new()  
    }  
}
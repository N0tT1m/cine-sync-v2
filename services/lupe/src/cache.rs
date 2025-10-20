use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct UserPreferences {
    pub preferred_genres: Option<Vec<String>>,
    pub request_count: u64,
    pub last_request_time: Option<DateTime<Utc>>,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            preferred_genres: None,
            request_count: 0,
            last_request_time: None,
        }
    }
}

#[derive(Clone)]
pub struct UserCache {
    cache: Arc<DashMap<u64, UserPreferences>>,
}

impl UserCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
        }
    }

    pub fn get_preferences(&self, user_id: u64) -> Option<UserPreferences> {
        self.cache.get(&user_id).map(|entry| entry.clone())
    }

    pub fn set_preferences(&self, user_id: u64, preferences: UserPreferences) {
        self.cache.insert(user_id, preferences);
    }

    pub fn update_request_count(&self, user_id: u64) {
        let mut entry = self.cache.entry(user_id).or_insert_with(UserPreferences::default);
        entry.request_count += 1;
        entry.last_request_time = Some(Utc::now());
    }

    pub fn get_user_count(&self) -> usize {
        self.cache.len()
    }

    pub fn clear(&self) {
        self.cache.clear();
    }
}

impl Default for UserCache {
    fn default() -> Self {
        Self::new()
    }
}
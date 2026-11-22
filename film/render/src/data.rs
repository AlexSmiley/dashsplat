//! Project tables compiled into the film (exported by viz/export_data.py from the project's own data).
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
pub struct Drive {
    pub t: Vec<f32>,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub yaw: Vec<f32>,
    pub speed: Vec<f32>,
    pub dist: Vec<f32>,
    pub duration: f32,
    pub length_m: f32,
}

#[derive(Deserialize)]
pub struct Commit {
    pub time: String,
    pub minute: f32,
    pub subject: String,
}

#[derive(Deserialize)]
pub struct Mover {
    pub id: u32,
    pub cls: String,
    pub size: Vec<f32>,
    pub t: Vec<f32>,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub yaw: Vec<f32>,
    pub state: Vec<i32>,
}

impl Mover {
    /// (x, y, yaw, predicted) at drive time `t`, or None outside the track
    pub fn at(&self, t: f32) -> Option<(f32, f32, f32, bool)> {
        if self.t.is_empty() || t < self.t[0] || t > *self.t.last().unwrap() {
            return None;
        }
        if self.t.len() == 1 {
            return Some((self.x[0], self.y[0], self.yaw[0], false));
        }
        let i = self.t.partition_point(|&v| v <= t).clamp(1, self.t.len() - 1);
        let a = ((t - self.t[i - 1]) / (self.t[i] - self.t[i - 1]).max(1e-4)).clamp(0.0, 1.0);
        let mut dy = self.yaw[i] - self.yaw[i - 1];
        if dy > std::f32::consts::PI { dy -= 2.0 * std::f32::consts::PI }
        if dy < -std::f32::consts::PI { dy += 2.0 * std::f32::consts::PI }
        Some((self.x[i - 1] + (self.x[i] - self.x[i - 1]) * a, self.y[i - 1] + (self.y[i] - self.y[i - 1]) * a,
              self.yaw[i - 1] + dy * a, self.state[i - 1] == 1))
    }
}

#[derive(Deserialize)]
pub struct Ground {
    pub bounds: [f32; 4],
    pub mpp: f32,
    pub size: [f32; 2],
}

pub struct Data {
    pub drive: Drive,
    pub commits: Vec<Commit>,
    pub movers: Vec<Mover>,
    pub ground: Ground,
}

pub fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| Data {
        drive: serde_json::from_str(include_str!("../data/drive.json")).expect("drive.json"),
        commits: serde_json::from_str(include_str!("../data/commits.json")).expect("commits.json"),
        movers: serde_json::from_str(include_str!("../data/movers.json")).expect("movers.json"),
        ground: serde_json::from_str(include_str!("../data/ground.json")).expect("ground.json"),
    })
}

impl Drive {
    /// Index of the last sample at or before drive time `t`.
    pub fn index_at(&self, t: f32) -> usize {
        match self.t.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
    }
    /// Drive time at which the car has covered `m` metres (the last moment of any stop before it).
    pub fn t_at_dist(&self, m: f32) -> f32 {
        let i = self.dist.partition_point(|&v| v < m).min(self.t.len() - 1);
        if i == 0 { return self.t[0]; }
        let (d0, d1) = (self.dist[i - 1], self.dist[i]);
        let a = if d1 > d0 { ((m - d0) / (d1 - d0)).clamp(0.0, 1.0) } else { 1.0 };
        self.t[i - 1] + (self.t[i] - self.t[i - 1]) * a
    }
    /// Interpolated (x, y, yaw, speed) at drive time `t`.
    pub fn at(&self, t: f32) -> (f32, f32, f32, f32) {
        let i = self.index_at(t).min(self.t.len() - 2);
        let a = ((t - self.t[i]) / (self.t[i + 1] - self.t[i])).clamp(0.0, 1.0);
        let l = |v: &Vec<f32>| v[i] + (v[i + 1] - v[i]) * a;
        let mut dy = self.yaw[i + 1] - self.yaw[i];
        if dy > std::f32::consts::PI { dy -= 2.0 * std::f32::consts::PI }
        if dy < -std::f32::consts::PI { dy += 2.0 * std::f32::consts::PI }
        (l(&self.x), l(&self.y), self.yaw[i] + dy * a, l(&self.speed))
    }
}

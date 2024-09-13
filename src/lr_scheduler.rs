use candle_nn::{AdamW, Optimizer};

pub struct WarmupLRScheduler {
    initial_lr: f64,
    decay_rate: f64,
    warmup_steps: usize,
    step: usize,
}

impl WarmupLRScheduler {
    pub fn new(initial_lr: f64, decay_rate: f64, warmup_steps: usize) -> Self {
        Self {
            initial_lr,
            decay_rate,
            warmup_steps,
            step: 0,
        }
    }

    pub fn step(&mut self) -> f64 {
        self.step += 1;
        if self.step <= self.warmup_steps {
            // Linear warm-up
            self.initial_lr * (self.step as f64 / self.warmup_steps as f64)
        } else {
            // Exponential decay after warm-up
            self.initial_lr * self.decay_rate.powf((self.step - self.warmup_steps) as f64)
        }
    }

    pub fn apply(&mut self, optimizer: &mut AdamW) {
        let new_lr = self.step();
        optimizer.set_learning_rate(new_lr);
    }
}

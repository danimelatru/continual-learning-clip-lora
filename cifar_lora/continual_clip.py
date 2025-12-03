import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_CHECKPOINT = "openai/clip-vit-base-patch32" # We use the visual encoder of CLIP
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

# --- 1. DATA PREPARATION (SPLIT CIFAR-10) ---
print_section("Loading and Splitting Data")
dataset = load_dataset("cifar10")

# Split classes: Task A (0-4) and Task B (5-9)
task_a_indices = [i for i, label in enumerate(dataset['train']['label']) if label < 5]
task_b_indices = [i for i, label in enumerate(dataset['train']['label']) if label >= 5]
test_a_indices = [i for i, label in enumerate(dataset['test']['label']) if label < 5]

# Create subsets
train_ds_a = dataset['train'].select(task_a_indices)
train_ds_b = dataset['train'].select(task_b_indices)
test_ds_a = dataset['test'].select(test_a_indices) # We only care about remembering A

print(f"Task A (Classes 0-4): {len(train_ds_a)} training samples")
print(f"Task B (Classes 5-9): {len(train_ds_b)} training samples")

# Image Preprocessing
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

def transform(example_batch):
    key = 'image' if 'image' in example_batch else 'img'
    
    inputs = processor([x for x in example_batch[key]], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

train_ds_a.set_transform(transform)
train_ds_b.set_transform(transform)
test_ds_a.set_transform(transform)

# --- HELPER FUNCTION: CUSTOM COLLATOR ---
# This prevents the Trainer from adding text-related keys (input_ids) that crash CLIP
def collate_fn(examples):
    return {
        'pixel_values': torch.stack([example['pixel_values'] for example in examples]),
        'labels': torch.tensor([example['labels'] for example in examples])
    }

# --- HELPER FUNCTION: EVALUATION ---
def evaluate_model(model, dataset, name):
    print(f"Evaluating on {name}...")
    
    # We must use remove_unused_columns=False to prevent deletion of image data
    args_eval = TrainingArguments(
        output_dir="./eval_results", 
        remove_unused_columns=False 
    )
    
    # We also use the custom collator here just in case
    trainer = Trainer(model=model, args=args_eval, data_collator=collate_fn)
    
    eval_result = trainer.evaluate(dataset) 
    print(f"Accuracy/Loss on {name}: {eval_result}")
    return eval_result

# --- EXPERIMENT 1: NAIVE FINE-TUNING (The "Problem") ---
print_section("EXPERIMENT 1: Full Fine-Tuning (The Baseline)")

# Load Clean Model
model_ft = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=10, 
    ignore_mismatched_sizes=True
)

# Step 1: Train on Task A
print("Training on Task A...")
args_a = TrainingArguments(
    output_dir="./results_ft_A",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=50,
    save_strategy="no",
    remove_unused_columns=False
)
# For standard Fine-Tuning, default collator usually works, but custom is safer
trainer_ft = Trainer(model=model_ft, args=args_a, train_dataset=train_ds_a, data_collator=collate_fn)
trainer_ft.train()

# Evaluate on A (Should be high)
acc_after_a = evaluate_model(model_ft, test_ds_a, "Test A (After Training A)")

# Step 2: Train on Task B (The source of forgetting)
print("Training on Task B (Continual Step)...")
args_b = TrainingArguments(
    output_dir="./results_ft_B",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=50,
    save_strategy="no",
    remove_unused_columns=False
)
trainer_ft = Trainer(model=model_ft, args=args_b, train_dataset=train_ds_b, data_collator=collate_fn)
trainer_ft.train()

# Evaluate on A again (Should be LOW -> Catastrophic Forgetting)
acc_after_b = evaluate_model(model_ft, test_ds_a, "Test A (After Training B)")


# --- EXPERIMENT 2: LoRA (The "Solution") ---
print_section("EXPERIMENT 2: PEFT / LoRA")

# Load Clean Model Again
model_lora = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=10, 
    ignore_mismatched_sizes=True
)

# Apply LoRA Config
peft_config = LoraConfig(
    inference_mode=False, 
    r=16,           
    lora_alpha=16, 
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"] 
)

model_lora = get_peft_model(model_lora, peft_config)
model_lora.print_trainable_parameters() 

# Step 1: Train LoRA on Task A
print("Training LoRA on Task A...")
# We must use collate_fn HERE to avoid 'input_ids' error
trainer_lora = Trainer(
    model=model_lora, 
    args=args_a, 
    train_dataset=train_ds_a, 
    data_collator=collate_fn
)
trainer_lora.train()

# Step 2: Train LoRA on Task B
print("Training LoRA on Task B...")
# We must use collate_fn HERE to avoid 'input_ids' error
trainer_lora = Trainer(
    model=model_lora, 
    args=args_b, 
    train_dataset=train_ds_b, 
    data_collator=collate_fn
)
trainer_lora.train()

# Evaluate on A again
acc_lora_after_b = evaluate_model(model_lora, test_ds_a, "Test A (LoRA - After Training B)")

# --- RESULTS SUMMARY & PLOTTING ---
print_section("FINAL RESULTS & PLOTTING")

loss_ideal = acc_after_a['eval_loss']
loss_ft = acc_after_b['eval_loss']
loss_lora = acc_lora_after_b['eval_loss']

print(f"Baseline - Accuracy on Task A after learning B: {loss_ft:.4f}")
print(f"LoRA     - Accuracy on Task A after learning B: {loss_lora:.4f}")

# Generate Graph
print("Generating comparison graph...")
labels = ['Ideal (Initial)', 'Fine-Tuning (Forgot)', 'LoRA (Mitigated)']
values = [loss_ideal, loss_ft, loss_lora]
colors = ['lightgray', 'salmon', 'lightgreen']

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Loss (Lower is Better)')
plt.title('Catastrophic Forgetting Mitigation: LoRA vs Full Fine-Tuning')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save figure
filename = "comparison_graph.png"
plt.savefig(filename)
print(f"Graph saved as {filename}")
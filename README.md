Medical Prescription OCR Pipeline - Development Journey
Overview

This project tackles one of the most challenging problems in healthcare digitization: extracting structured information from handwritten doctor prescriptions. This README documents the complete journey from initial experimentation to the final production-ready solution.

Note: The final solution uses LLM-based OCR, but this was chosen after extensive experimentation with multiple traditional approaches that proved insufficient for real-world medical handwriting.

The Challenge

Handwritten medical prescriptions present unique difficulties:

Highly variable and often illegible handwriting styles

Medical terminology and abbreviations (OD, BD, TID, mg, ml, etc.)

Noisy image quality with varying lighting and contrast

Overlapping text and irregular baselines

Critical accuracy requirements (errors can have serious consequences)

Development Journey: Three Major Approaches
⚙️ Method 1: Custom HTR (Handwritten Text Recognition) Pipeline

Architecture Components:

Document Layout Analysis (DLA)

Noise Reduction & Normalization

HTR Models (Transformers, PARSeq, TrOCR, CRNNs)

Named Entity Recognition (NER)

Key-Value Pair Extraction (KVPE)

Why HTR?

HTR models are specifically designed for handwriting recognition, unlike general OCR. Modern HTR architectures can handle:

Different handwriting styles and stroke-level variations

Broken characters and cursive scripts

Doctor’s freeflow handwriting patterns

Implementation Details:

Tested multiple architectures: Transformers, PARSeq, TrOCR, CRNNs

Implemented stroke-level character recognition

Built custom preprocessing pipeline for medical documents

Results:

Word Accuracy: 75–90% (dataset dependent)

Good for general handwriting but struggled with medical terminology

Critical Limitations:

❌ Required 10k+ handwritten images for adequate training
❌ Sensitive to writing angle variance and irregular baselines
❌ No inherent medical vocabulary constraints → frequent hallucinations
❌ Failed on extremely cursive or noisy prescriptions
❌ Could not reliably distinguish similar medical terms

Verdict: Insufficient accuracy for production medical use cases.

⚙️ Method 2: CTC-Based Recognition Pipeline

Architecture Components:

Document Layout Analysis (DLA)

Noise Reduction & Normalization

CTC (Connectionist Temporal Classification) Loss

Named Entity Recognition (NER)

Key-Value Pair Extraction (KVPE)

Approach:

Used CTC loss function for sequence-to-sequence learning without explicit alignment between input and output sequences.

Results:

Similar accuracy range to Method 1 with comparable limitations.

Limitations:

❌ Still required large training datasets
❌ No contextual understanding of medical terminology
❌ Struggled with domain-specific patterns

Verdict: Not a significant improvement over Method 1.

⚙️ Method 3: Transfer Learning with Fine-Tuned Models

Architecture Components:

Document Layout Analysis (DLA)

Noise Reduction & Normalization

Transfer Learning on pretrained models

Named Entity Recognition (NER)

Key-Value Pair Extraction (KVPE)

Base Models Tested:

TrOCR

Donut (Document Understanding Transformer)

PARSeq

LayoutLMv3

ViTSTR

Why Transfer Learning?

Pretrained models already understand:

Character shapes and attention over text lines

Multi-language character sets

Context-aware decoding

Through fine-tuning, models learn:

Medical vocabulary (drug names)

Dosage format patterns (500mg, 1gm, etc.)

Doctor’s shorthand symbols

Prescription-specific document structure

Custom Tokenization Strategy:

Implemented SentencePiece (Unigram/BPE) with custom medical vocabulary:

Works without whitespace (critical for handwriting)

Learns subword units matching drug names

Handles spelling variations

Eliminates Out-of-Vocabulary (OOV) issues

Tokenization Settings:

Vocab size: 2,000–6,000 tokens
Model type: Unigram
Medical tokens: mg, ml, OD, BD, TID, SOS, Tab, Cap, Inj., 1-0-1, 0-1-1, 500mg, 1gm


Example:

"Azithromycin 500mg OD" →
Tokens: ▁Azi, thro, mycin, ▁500, mg, ▁OD

Implementation 3A: Custom Training Pipeline (Kaggle-Optimized)

Key Components:

Vision Encoder: ResNet18 adapted for handwriting images

Transformer Decoder: Autoregressive text generation

Training Infrastructure:

Teacher forcing

AdamW + cosine schedule

Gradient clipping

Validation metrics

Inference: Greedy decoding

Preprocessing Pipeline:

Grayscale conversion

Gaussian blur + CLAHE

Pixel normalization

Aspect-ratio-preserving padding

Results:

Word Accuracy: 80–95%

Strong performance on clean handwriting

Better context understanding

Implementation 3B: Pretrained Model Fine-Tuning

Tested pretrained: chinmays18/medical-prescription-ocr (Donut-based)

Limitations of Both Transfer Learning Approaches:

❌ Struggled with highly variable doctor handwriting
❌ Hallucination issues
❌ Required significant labeled data
❌ High compute cost
❌ Overfitting risk
❌ Still not production ready

Verdict: Better than Methods 1–2, but still insufficient for real medical use.

🎯 Final Solution: LLM-Based Vision OCR Pipeline

After testing all classical approaches, Vision-Language Models gave breakthrough accuracy.

Why LLMs Succeeded

Pre-trained on massive diverse handwriting

Contextual understanding

Built-in medical knowledge

Structured JSON output

Zero-shot generalization

✅ Production Pipeline Architecture
Pipeline Flow:
Input Image → Preprocessing → Vision LLM OCR → Structured Extraction → Output Generation

1. Pre-Processing Module (OpenCV)

Steps:

Deskew

Denoise

Contrast enhance using CLAHE

Output: Cleaned image ready for OCR

2. OCR Extraction (Groq Vision LLM)

Model: meta-llama/llama-4-scout-17b-16e-instruct

Process:

Convert image to Base64

Send to multimodal LLM

Receive structured JSON extraction

Extracted Fields:

Raw transcription

Patient information

Doctor’s notes

Medications + dosages

Dates

All PII fields

3. Text Cleaning

Normalize spacing

Fix common OCR errors

Remove noise symbols

4. Structured Data Assembly

PatientInfo

MedicalNote

ExtractionResult

5. Multi-Format Output Generation
Output Type	Description
_structured.json	Full structured extraction
_report.txt	Human-readable summary
_preprocessed.jpg	Cleaned image
_redacted.jpg	Privacy redacted
Console Summary	Quick overview
6. PII Redaction

Automatic PII detection

Blackout regions

HIPAA-compliant outputs

Performance Comparison
Approach	Word Accuracy	Medical Term Accuracy	Training Data Required	Production Ready
Method 1 (HTR)	75–90%	Low	10k+	❌
Method 2 (CTC)	75–85%	Low	10k+	❌
Method 3 (Transfer)	80–95%	Medium	5k+	❌
Final (LLM Vision)	95–99%	High	Zero-shot	✅
Key Learnings

Traditional ML approaches require huge labeled datasets

Fine-tuning helps but is expensive and unstable

Vision-Language Models generalize exceptionally well

Preprocessing strongly affects outcome

Structured output is essential for healthcare workflows

Technical Stack

Traditional (Methods 1–3):

PyTorch, TensorFlow

OpenCV

Transformers

ResNet, TrOCR, Donut

SentencePiece

Final Solution:

OpenCV

Groq Vision API

Python

JSON Schema

Conclusion

This project demonstrates that selecting the right tool matters more than model complexity. After multiple advanced pipelines, LLM vision models finally delivered:

Highest accuracy

Zero-shot performance

Real-world reliability

Lower development + maintenance cost

Future Work

Multi-page support

Faster inference

EHR integration

Multilingual prescriptions

Advanced PII detection

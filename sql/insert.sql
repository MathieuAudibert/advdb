INSERT INTO Company (name) VALUES ('OpenAI');
INSERT INTO Company (name) VALUES ('Anthropic');
INSERT INTO Company (name) VALUES ('Google');
INSERT INTO Company (name) VALUES ('Midjourney');
INSERT INTO Company (name) VALUES ('Stability AI');
INSERT INTO Company (name) VALUES ('Runway');
INSERT INTO Company (name) VALUES ('Pika');
INSERT INTO Company (name) VALUES ('Suno');
INSERT INTO Company (name) VALUES ('Udio');
INSERT INTO Company (name) VALUES ('ElevenLabs');
INSERT INTO Company (name) VALUES ('GitHub');
INSERT INTO Company (name) VALUES ('Anysphere');
INSERT INTO Company (name) VALUES ('Sourcegraph');
INSERT INTO Company (name) VALUES ('Amazon Web Services');
INSERT INTO Company (name) VALUES ('JetBrains');
INSERT INTO Company (name) VALUES ('Perplexity AI');
INSERT INTO Company (name) VALUES ('Meta');
INSERT INTO Company (name) VALUES ('Mistral AI');
INSERT INTO Company (name) VALUES ('Pinecone');
INSERT INTO Company (name) VALUES ('Weaviate');
INSERT INTO Company (name) VALUES ('Zilliz');
INSERT INTO Company (name) VALUES ('Qdrant');
INSERT INTO Company (name) VALUES ('Community');
INSERT INTO Company (name) VALUES ('LangChain');
INSERT INTO Company (name) VALUES ('LlamaIndex');
INSERT INTO Company (name) VALUES ('deepset');
INSERT INTO Company (name) VALUES ('Deepgram');
INSERT INTO Company (name) VALUES ('AssemblyAI');
INSERT INTO Company (name) VALUES ('Google Cloud');
INSERT INTO Company (name) VALUES ('NVIDIA');
INSERT INTO Company (name) VALUES ('Lakera');
INSERT INTO Company (name) VALUES ('Guardrails AI');
INSERT INTO Company (name) VALUES ('ETH SRI');
INSERT INTO Company (name) VALUES ('OpenRouter');
INSERT INTO Company (name) VALUES ('Replicate');
INSERT INTO Company (name) VALUES ('Modal Labs');
INSERT INTO Company (name) VALUES ('Vercel');
INSERT INTO Company (name) VALUES ('Figma');
INSERT INTO Company (name) VALUES ('Framer');
INSERT INTO Company (name) VALUES ('Canva');
INSERT INTO Company (name) VALUES ('Notion');
INSERT INTO Company (name) VALUES ('Cohere');
INSERT INTO Company (name) VALUES ('xAI');
INSERT INTO Company (name) VALUES ('DeepSeek');
INSERT INTO Company (name) VALUES ('Alibaba Cloud');
INSERT INTO Company (name) VALUES ('Databricks');
INSERT INTO Company (name) VALUES ('Microsoft');
INSERT INTO Company (name) VALUES ('MosaicML');
INSERT INTO Company (name) VALUES ('Vanna AI');
INSERT INTO Company (name) VALUES ('Dataherald');
INSERT INTO Company (name) VALUES ('Hugging Face');
INSERT INTO Company (name) VALUES ('Comfy Org');
INSERT INTO Company (name) VALUES ('Krea');
INSERT INTO Company (name) VALUES ('Leonardo.Ai');
INSERT INTO Company (name) VALUES ('Ideogram');
INSERT INTO Company (name) VALUES ('Black Forest Labs');
INSERT INTO Company (name) VALUES ('Playground');
INSERT INTO Company (name) VALUES ('Luma AI');
INSERT INTO Company (name) VALUES ('Kuaishou');
INSERT INTO Company (name) VALUES ('HeyGen');
INSERT INTO Company (name) VALUES ('Synthesia');
INSERT INTO Company (name) VALUES ('ByteDance');
INSERT INTO Company (name) VALUES ('Descript');
INSERT INTO Company (name) VALUES ('PlayHT');
INSERT INTO Company (name) VALUES ('Coqui');
INSERT INTO Company (name) VALUES ('Voicemod');
INSERT INTO Company (name) VALUES ('Adobe');
INSERT INTO Company (name) VALUES ('Tome');
INSERT INTO Company (name) VALUES ('Gamma');
INSERT INTO Company (name) VALUES ('Beautiful.ai');
INSERT INTO Company (name) VALUES ('SlidesAI');
INSERT INTO Company (name) VALUES ('Blackbox');
INSERT INTO Company (name) VALUES ('Tabnine');
INSERT INTO Company (name) VALUES ('Codeium (Qodo)');
INSERT INTO Company (name) VALUES ('Replit');
INSERT INTO Company (name) VALUES ('Microsoft Azure');
INSERT INTO Company (name) VALUES ('Arize AI');
INSERT INTO Company (name) VALUES ('Ollama');
INSERT INTO Company (name) VALUES ('LM Studio');
INSERT INTO Company (name) VALUES ('Silero');
INSERT INTO Company (name) VALUES ('Papers with Code');
INSERT INTO Company (name) VALUES ('Kaggle');
INSERT INTO Company (name) VALUES ('OpenHands');

INSERT INTO IA_Type (category, modality) VALUES ('LLMs & Chat Assistants', 'multimodal');
INSERT INTO IA_Type (category, modality) VALUES ('Image Gen & Editing', 'image');
INSERT INTO IA_Type (category, modality) VALUES ('Video Gen & Editing', 'video');
INSERT INTO IA_Type (category, modality) VALUES ('Video Gen & Editing', 'audio');
INSERT INTO IA_Type (category, modality) VALUES ('Audio/Music/TTS', 'audio');
INSERT INTO IA_Type (category, modality) VALUES ('LLMs & Chat Assistants', 'code');
INSERT INTO IA_Type (category, modality) VALUES ('Code Assistants', 'code');
INSERT INTO IA_Type (category, modality) VALUES ('Other', 'multimodal');
INSERT INTO IA_Type (category, modality) VALUES ('Search & RAG', 'infra');
INSERT INTO IA_Type (category, modality) VALUES ('Other', 'code');
INSERT INTO IA_Type (category, modality) VALUES ('Other', 'audio');
INSERT INTO IA_Type (category, modality) VALUES ('Safety & Guardrails', 'safety');
INSERT INTO IA_Type (category, modality) VALUES ('Infra & Inference', 'infra');
INSERT INTO IA_Type (category, modality) VALUES ('Design & UI', 'productivity');
INSERT INTO IA_Type (category, modality) VALUES ('Design & UI', 'design');
INSERT INTO IA_Type (category, modality) VALUES ('Other', 'productivity');
INSERT INTO IA_Type (category, modality) VALUES ('Productivity & Copilots', 'productivity');
INSERT INTO IA_Type (category, modality) VALUES ('LLMs & Chat Assistants', 'text');
INSERT INTO IA_Type (category, modality) VALUES ('Other', 'infra');
INSERT INTO IA_Type (category, modality) VALUES ('Evaluation & Benchmarks', 'infra');
INSERT INTO IA_Type (category, modality) VALUES ('Speech-to-Text (ASR)', 'audio');
INSERT INTO IA_Type (category, modality) VALUES ('Other', 'image');
INSERT INTO IA_Type (category, modality) VALUES ('Video Gen & Editing', 'image');
INSERT INTO IA_Type (category, modality) VALUES ('Video Gen & Editing', 'code');
INSERT INTO IA_Type (category, modality) VALUES ('Safety & Guardrails', 'text');

INSERT INTO Config (api_available, open_source) VALUES (1, 0);
INSERT INTO Config (api_available, open_source) VALUES (0, 0);
INSERT INTO Config (api_available, open_source) VALUES (1, 1);
INSERT INTO Config (api_available, open_source) VALUES (0, 1);

INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0);
INSERT INTO Specs (mod_text, mod_image, mod_video, mod_audio, mod_code, mod_design, mod_infra, mod_productivity, mod_safety, mod_multimodal, modality_count) VALUES (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('ChatGPT', 'https://chatgpt.com', TO_DATE('2022', 'YYYY'), 1, 1, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Claude', 'https://claude.ai', TO_DATE('2023', 'YYYY'), 1, 2, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Gemini', 'https://gemini.google.com', TO_DATE('2023', 'YYYY'), 1, 3, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Midjourney', 'https://www.midjourney.com', TO_DATE('2022', 'YYYY'), 2, 4, 2, 2);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Stable Diffusion', 'https://stability.ai/stable-image', TO_DATE('2022', 'YYYY'), 2, 5, 2, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('DALL·E 3', 'https://openai.com/index/dall-e-3-system-card', TO_DATE('2023', 'YYYY'), 2, 1, 2, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Runway Gen-3', 'https://runwayml.com', TO_DATE('2024', 'YYYY'), 3, 6, 3, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Pika', 'https://pika.art', TO_DATE('2023', 'YYYY'), 3, 7, 3, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Suno', 'https://suno.com', TO_DATE('2023', 'YYYY'), 4, 8, 4, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Udio', 'https://www.udio.com', TO_DATE('2024', 'YYYY'), 4, 9, 4, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('ElevenLabs', 'https://elevenlabs.io', TO_DATE('2022', 'YYYY'), 4, 10, 5, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('GitHub Copilot', 'https://github.com/features/copilot', TO_DATE('2022', 'YYYY'), 5, 11, 6, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Cursor', 'https://cursor.com', TO_DATE('2023', 'YYYY'), 5, 12, 7, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Sourcegraph Cody', 'https://sourcegraph.com/cody', TO_DATE('2023', 'YYYY'), 5, 13, 6, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Amazon Q Developer', 'https://aws.amazon.com/q/developer/', TO_DATE('2023', 'YYYY'), 5, 14, 6, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('JetBrains AI Assistant', 'https://www.jetbrains.com/ai-assistant/', TO_DATE('2023', 'YYYY'), 5, 15, 6, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Perplexity', 'https://www.perplexity.ai', TO_DATE('2022', 'YYYY'), 1, 16, 8, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Meta AI', 'https://www.meta.ai', TO_DATE('2024', 'YYYY'), 1, 17, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Mistral Le Chat', 'https://chat.mistral.ai', TO_DATE('2024', 'YYYY'), 1, 18, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Pinecone', 'https://www.pinecone.io', TO_DATE('2021', 'YYYY'), 6, 19, 9, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Weaviate', 'https://weaviate.io', TO_DATE('2019', 'YYYY'), 6, 20, 9, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Milvus', 'https://milvus.io', TO_DATE('2019', 'YYYY'), 6, 21, 9, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Qdrant', 'https://qdrant.tech', TO_DATE('2021', 'YYYY'), 6, 22, 9, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('pgvector', 'https://github.com/pgvector/pgvector', TO_DATE('2023', 'YYYY'), 6, 23, 9, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('LangChain', 'https://www.langchain.com', TO_DATE('2022', 'YYYY'), 5, 24, 10, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('LlamaIndex', 'https://www.llamaindex.ai', TO_DATE('2022', 'YYYY'), 5, 25, 10, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Haystack', 'https://haystack.deepset.ai', TO_DATE('2024', 'YYYY'), 5, 26, 10, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Whisper', 'https://openai.com/index/whisper/', TO_DATE('2022', 'YYYY'), 4, 1, 11, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Deepgram', 'https://deepgram.com', TO_DATE('2023', 'YYYY'), 4, 27, 5, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('AssemblyAI', 'https://www.assemblyai.com', TO_DATE('2017', 'YYYY'), 4, 28, 11, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Google Speech-to-Text', 'https://cloud.google.com/speech-to-text', TO_DATE('2018', 'YYYY'), 4, 29, 11, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('NVIDIA NeMo Guardrails', 'https://developer.nvidia.com/nemo-guardrails', TO_DATE('2023', 'YYYY'), 7, 30, 12, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Lakera Guard', 'https://www.lakera.ai/lakera-guard', TO_DATE('2023', 'YYYY'), 7, 31, 12, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Guardrails AI (framework)', 'https://www.guardrailsai.com/', TO_DATE('2023', 'YYYY'), 7, 32, 12, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('LMQL', 'https://lmql.ai/', TO_DATE('2023', 'YYYY'), 5, 33, 6, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('OpenRouter', 'https://openrouter.ai', TO_DATE('2023', 'YYYY'), 6, 34, 19, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Replicate', 'https://replicate.com', TO_DATE('2021', 'YYYY'), 6, 35, 13, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Modal', 'https://modal.com', TO_DATE('2022', 'YYYY'), 6, 36, 13, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Vercel v0', 'https://v0.app', TO_DATE('2023', 'YYYY'), 8, 37, 14, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Figma AI', 'https://www.figma.com/ai/', TO_DATE('2024', 'YYYY'), 9, 38, 15, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Framer AI', 'https://www.framer.com/features/ai/', TO_DATE('2023', 'YYYY'), 9, 39, 15, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Canva Magic Write', 'https://www.canva.com/magic-write/', TO_DATE('2022', 'YYYY'), 8, 40, 16, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Notion AI', 'https://www.notion.com/product/ai', TO_DATE('2023', 'YYYY'), 8, 41, 17, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Cohere Command R', 'https://cohere.com', TO_DATE('2024', 'YYYY'), 1, 42, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('xAI Grok', 'https://x.ai', TO_DATE('2023', 'YYYY'), 1, 43, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('DeepSeek R1', 'https://www.deepseek.com', TO_DATE('2025', 'YYYY'), 1, 44, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Mistral Large', 'https://mistral.ai', TO_DATE('2023', 'YYYY'), 1, 18, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Llama 3.1', 'https://ai.meta.com/llama/', TO_DATE('2023', 'YYYY'), 1, 17, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Qwen 2.5', 'https://qwenlm.ai/', TO_DATE('2025', 'YYYY'), 1, 45, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Gemma 2', 'https://ai.google.dev/gemma', TO_DATE('2024', 'YYYY'), 1, 3, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('DBRX', 'https://www.databricks.com/research/dbrx', TO_DATE('2024', 'YYYY'), 1, 46, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Phi-3', 'https://www.microsoft.com/ai', TO_DATE('2024', 'YYYY'), 1, 47, 1, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('MPT-7B', 'https://www.mosaicml.com', TO_DATE('2023', 'YYYY'), 10, 48, 18, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Mixtral 8x7B', 'https://mistral.ai', TO_DATE('2023', 'YYYY'), 10, 18, 18, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('OpenAI o3-mini', 'https://platform.openai.com', TO_DATE('2025', 'YYYY'), 1, 1, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('GPT-4o', 'https://platform.openai.com', TO_DATE('2023', 'YYYY'), 1, 1, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Claude 3.7 Sonnet', 'https://www.anthropic.com', TO_DATE('2023', 'YYYY'), 1, 2, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Grok-2', 'https://x.ai', TO_DATE('2024', 'YYYY'), 1, 43, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Mistral Codestral', 'https://mistral.ai', TO_DATE('2023', 'YYYY'), 5, 18, 6, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Code Llama', 'https://ai.meta.com/llama/', TO_DATE('2023', 'YYYY'), 5, 17, 6, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('OpenAI o4-mini', 'https://platform.openai.com', TO_DATE('2025', 'YYYY'), 1, 1, 1, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Llama Guard', 'https://ai.meta.com', TO_DATE('2023', 'YYYY'), 7, 17, 12, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Vanna', 'https://www.vanna.ai', TO_DATE('2024', 'YYYY'), 8, 49, 16, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Dataherald', 'https://www.dataherald.com', TO_DATE('2023', 'YYYY'), 8, 50, 16, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Transformers', 'https://huggingface.co/docs/transformers', TO_DATE('2018', 'YYYY'), 5, 51, 10, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Diffusers', 'https://huggingface.co/docs/diffusers', TO_DATE('2022', 'YYYY'), 2, 51, 22, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Gradio', 'https://www.gradio.app', TO_DATE('2019', 'YYYY'), 8, 51, 16, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Hugging Face Spaces', 'https://huggingface.co/spaces', TO_DATE('2021', 'YYYY'), 6, 51, 13, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('ComfyUI', 'https://github.com/comfyanonymous/ComfyUI', TO_DATE('2023', 'YYYY'), 2, 52, 2, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Automatic1111', 'https://github.com/AUTOMATIC1111/stable-diffusion-webui', TO_DATE('2022', 'YYYY'), 2, 23, 2, 4);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Krea AI', 'https://www.krea.ai', TO_DATE('2022', 'YYYY'), 2, 53, 2, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Leonardo AI', 'https://leonardo.ai', TO_DATE('2022', 'YYYY'), 2, 54, 2, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Ideogram', 'https://ideogram.ai', TO_DATE('2023', 'YYYY'), 2, 55, 2, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Black Forest Labs FLUX', 'https://blackforestlabs.ai', TO_DATE('2024', 'YYYY'), 2, 56, 2, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Playground AI', 'https://playground.com', TO_DATE('2022', 'YYYY'), 2, 57, 2, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Luma Dream Machine', 'https://lumalabs.ai', TO_DATE('2024', 'YYYY'), 3, 58, 3, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Kling AI', 'https://klingai.com', TO_DATE('2024', 'YYYY'), 3, 59, 3, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('HeyGen', 'https://www.heygen.com', TO_DATE('2022', 'YYYY'), 3, 60, 3, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Synthesia', 'https://www.synthesia.io', TO_DATE('2022', 'YYYY'), 3, 61, 3, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('CapCut AI', 'https://www.capcut.com/ai', TO_DATE('2019', 'YYYY'), 3, 62, 3, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Descript', 'https://www.descript.com', TO_DATE('2019', 'YYYY'), 4, 63, 4, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('PlayHT', 'https://play.ht', TO_DATE('2017', 'YYYY'), 4, 64, 5, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Coqui TTS', 'https://coqui.ai', TO_DATE('2021', 'YYYY'), 4, 65, 5, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Voicemod', 'https://www.voicemod.net', TO_DATE('2014', 'YYYY'), 4, 66, 5, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('OpenAI TTS', 'https://platform.openai.com', TO_DATE('2023', 'YYYY'), 4, 1, 11, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('OpenAI Sora', 'https://openai.com/sora', TO_DATE('2024', 'YYYY'), 3, 1, 3, 2);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Adobe Firefly', 'https://www.adobe.com/products/firefly.html', TO_DATE('2023', 'YYYY'), 2, 67, 2, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Adobe Photoshop Generative Fill', 'https://www.adobe.com/products/photoshop.html', TO_DATE('2023', 'YYYY'), 2, 67, 2, 2);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Illustrator Text to Vector', 'https://www.adobe.com/products/illustrator.html', TO_DATE('2023', 'YYYY'), 2, 67, 23, 2);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Canva Magic Design', 'https://www.canva.com/magic-design/', TO_DATE('2013', 'YYYY'), 9, 40, 15, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Tome AI', 'https://tome.app', TO_DATE('2022', 'YYYY'), 8, 68, 16, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Gamma', 'https://gamma.app', TO_DATE('2023', 'YYYY'), 8, 69, 16, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Beautiful.ai', 'https://www.beautiful.ai', TO_DATE('2018', 'YYYY'), 8, 70, 16, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('SlidesAI.io', 'https://www.slidesai.io', TO_DATE('2022', 'YYYY'), 8, 71, 16, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Blackbox AI', 'https://www.useblackbox.io', TO_DATE('2024', 'YYYY'), 5, 72, 6, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Tabnine', 'https://www.tabnine.com', TO_DATE('2018', 'YYYY'), 5, 73, 6, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Codeium', 'https://www.codeium.com', TO_DATE('2022', 'YYYY'), 5, 74, 6, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Replit Ghostwriter', 'https://replit.com/site/ghostwriter', TO_DATE('2022', 'YYYY'), 5, 75, 6, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('OpenAI Assistants API', 'https://platform.openai.com/docs/assistants/overview', TO_DATE('2020', 'YYYY'), 5, 1, 10, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Anthropic Tool Use', 'https://docs.anthropic.com', TO_DATE('2024', 'YYYY'), 5, 2, 10, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Google Vertex AI', 'https://cloud.google.com/vertex-ai', TO_DATE('2021', 'YYYY'), 6, 29, 19, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Azure AI Studio', 'https://ai.azure.com', TO_DATE('2023', 'YYYY'), 6, 76, 19, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('AWS Bedrock', 'https://aws.amazon.com/bedrock/', TO_DATE('2023', 'YYYY'), 6, 14, 19, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('LangSmith', 'https://smith.langchain.com', TO_DATE('2023', 'YYYY'), 6, 24, 20, 1);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('TruLens', 'https://www.trulens.org', TO_DATE('2023', 'YYYY'), 6, 77, 20, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Ollama', 'https://ollama.com', TO_DATE('2023', 'YYYY'), 6, 78, 19, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('LM Studio', 'https://lmstudio.ai', TO_DATE('2023', 'YYYY'), 6, 79, 19, 2);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('WhisperX', 'https://github.com/m-bain/whisperX', TO_DATE('2022', 'YYYY'), 4, 23, 21, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Silero VAD', 'https://github.com/snakers4/silero-vad', TO_DATE('2020', 'YYYY'), 4, 80, 5, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Papers with Code SOTA', 'https://paperswithcode.com', TO_DATE('2018', 'YYYY'), 6, 81, 9, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('Kaggle Models', 'https://www.kaggle.com/models', TO_DATE('2023', 'YYYY'), 6, 82, 19, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('OpenHands', 'https://github.com/All-Hands-AI/OpenHands', TO_DATE('2024', 'YYYY'), 5, 83, 24, 3);

INSERT INTO IA_Gen (name, website, release_year, fk_specs, fk_company, fk_iatype, fk_cfg) 
VALUES ('LlamaGuard 2', 'https://ai.meta.com', TO_DATE('2023', 'YYYY'), 10, 17, 25, 3);

COMMIT;
# Text Document Embedding Generator
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/simonpierreboucher/llm-generate-function)](https://github.com/simonpierreboucher/llm-generate-function/issues)
[![GitHub Forks](https://img.shields.io/github/forks/simonpierreboucher/llm-generate-function)](https://github.com/simonpierreboucher/llm-generate-function/network)
[![GitHub Stars](https://img.shields.io/github/stars/simonpierreboucher/llm-generate-function)](https://github.com/simonpierreboucher/llm-generate-function/stargazers)

A Python tool for generating embeddings from text documents using multiple providers including OpenAI, Mistral AI, Voyage AI, and Cohere. It splits documents into configurable-sized chunks and generates embeddings for each chunk.

## Features

- Support for multiple embedding providers:
  - OpenAI
  - Mistral AI
  - Voyage AI
  - Cohere
- Multiple text file processing
- Configurable chunk sizing
- Document header management
- Multiple output formats (CSV, JSON, NPY)
- Error handling and retries
- YAML-based configuration

## Prerequisites

```bash
pip install openai tiktoken numpy pandas tqdm pyyaml requests
```

## Configuration

Create a `config.yaml` file with the following structure:

```yaml
api:
  provider:
    name: "openai"  # Options: "openai", "mistral", "voyage", "cohere"
    key: "your-api-key"
    model: "text-embedding-3-large"  # Model varies by provider
  max_retries: 3
  retry_delay: 2

paths:
  input_folder: "path/to/text/files"
  output_base: "output"

processing:
  chunk_sizes: [400, 800, 1200]
  header_lines: 2

output:
  formats:
    - csv
    - json
    - npy
```

### Configuration Parameters

#### API Provider Settings
- OpenAI
  - `model`: "text-embedding-3-large"
  - Key environment variable: `OPENAI_API_KEY`

- Mistral AI
  - `model`: "mistral-embed"
  - Key environment variable: `MISTRAL_API_KEY`

- Voyage AI
  - `model`: "voyage-large-2"
  - Key environment variable: `VOYAGE_API_KEY`

- Cohere
  - `model`: "embed-english-v3.0"
  - Key environment variable: `CO_API_KEY`

[Rest of the configuration parameters remain the same]

## Usage

1. Set up your API key for the chosen provider:
```bash
# For OpenAI
export OPENAI_API_KEY='your-api-key'
# For Mistral
export MISTRAL_API_KEY='your-api-key'
# For Voyage
export VOYAGE_API_KEY='your-api-key'
# For Cohere
export CO_API_KEY='your-api-key'
```

2. Configure your provider in `config.yaml`
3. Prepare your text files in the input directory
4. Run the script:
```bash
python embedding_generator.py
```

## Provider-Specific Features

### OpenAI
- High-quality embeddings
- Extensive model options
- Reliable API performance

### Mistral AI
- Competitive pricing
- Good performance for multiple languages
- Modern embedding models

### Voyage AI
- Specialized for specific use cases
- Competitive pricing
- Good documentation

### Cohere
- Multiple embedding types
- Classification-specific embeddings
- Extensive language support

[Output Structure section remains the same]

## Error Handling

- Provider-specific error handling
- Automatic retry on API failure
- Exponential backoff between attempts
- Error and warning logging
- Continues processing if a provider fails

## Methods Description

### Provider Classes

#### `OpenAIProvider`
Handles embedding generation using OpenAI's API.

#### `MistralProvider`
Manages embeddings through Mistral AI's API.

#### `VoyageProvider`
Processes embeddings using Voyage AI's API.

#### `CohereProvider`
Generates embeddings via Cohere's API.

[Rest of the methods description remains the same]

## Limitations

- Requires valid API key for chosen provider
- Different rate limits per provider
- Varying embedding dimensions between providers
- Provider-specific model limitations
- Processes .txt files only

## Best Practices

[Previous best practices, plus:]

4. **Provider Selection**
   - Choose provider based on your needs:
     - OpenAI for general purpose
     - Mistral for multilingual
     - Voyage for specialized cases
     - Cohere for classification

5. **API Management**
   - Monitor usage across providers
   - Consider provider-specific rate limits
   - Keep API keys secure

[Rest of the sections remain the same]

## Provider Comparison

| Provider | Strengths | Use Cases |
|----------|-----------|----------|
| OpenAI | High quality, reliable | General purpose |
| Mistral | Good multilingual support | International content |
| Voyage | Specialized features | Domain-specific |
| Cohere | Classification focus | Text classification |



2. Prepare your text files in the input directory

3. Run the script:
```bash
python embedding_generator.py
```

## Output Structure

For each configured chunk size, the script generates:

### CSV (`embeddings_results_{size}tok.csv`)
- filename: Source file name
- chunk_id: Chunk identifier
- text: Chunk content
- embedding: Embedding vector

### JSON (`chunks.json`)
```json
{
  "text": "chunk content",
  "embedding": "chunk id",
  "metadata": {
    "filename": "file name",
    "chunk_id": "chunk id"
  }
}
```

### NPY (`embeddings.npy`)
NumPy array containing all embedding vectors

## Error Handling

- Automatic retry on API failure
- Exponential backoff between attempts
- Error and warning logging
- Continues processing if a file fails

## Methods Description

### EmbeddingGenerator Class

#### `clean_text(text: str) -> str`
Cleans and normalizes text by removing extra whitespace and line breaks.

#### `split_into_chunks(text: str, max_tokens: int) -> List[str]`
Splits text into chunks while preserving headers and respecting token limits.

#### `get_embedding(text: str) -> Optional[List[float]]`
Retrieves embeddings from OpenAI API with error handling and retries.

#### `process_file(file_path: str, chunk_size: int) -> List[Dict[str, Any]]`
Processes a single file, generating chunks and embeddings.

#### `save_results(results: List[Dict[str, Any]], chunk_size: int) -> None`
Saves results in configured output formats.

## Limitations

- Requires valid OpenAI API key
- Processes .txt files only
- Chunk sizes must comply with API limits
- Rate limits based on OpenAI API tier

## Best Practices

1. **File Preparation**
   - Ensure text files are properly encoded (UTF-8)
   - Remove any binary or non-text content

2. **Configuration**
   - Adjust chunk sizes based on your needs
   - Configure appropriate retry settings
   - Set reasonable header line count

3. **Resource Management**
   - Monitor API usage
   - Consider rate limiting for large datasets
   - Backup output files regularly

## Troubleshooting

Common issues and solutions:

1. **API Errors**
   - Verify API key
   - Check API rate limits
   - Ensure network connectivity

2. **File Processing Issues**
   - Check file encoding
   - Verify file permissions
   - Ensure valid file content

3. **Output Errors**
   - Check disk space
   - Verify write permissions
   - Validate output directory structure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Support

For issues and feature requests, please create an issue in the repository.

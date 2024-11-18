import os
import re
import yaml
import tiktoken
import requests
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from tqdm import tqdm
from openai import OpenAI

class EmbeddingProvider(ABC):
    """Classe abstraite pour les providers d'embeddings"""
    
    @abstractmethod
    def get_embedding(self, text: Union[str, List[str]]) -> Optional[List[float]]:
        """Obtient l'embedding pour un texte donné"""
        pass

class OpenAIProvider(EmbeddingProvider):
    """Provider pour OpenAI"""
    
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_embedding(self, text: Union[str, List[str]]) -> Optional[List[float]]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Erreur OpenAI: {e}")
            return None

class MistralProvider(EmbeddingProvider):
    """Provider pour Mistral AI"""
    
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.mistral.ai/v1/embeddings"

    def get_embedding(self, text: Union[str, List[str]]) -> Optional[List[float]]:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "input": text if isinstance(text, list) else [text],
                "model": self.model,
                "encoding_format": "float"
            }
            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"Erreur Mistral: {e}")
            return None

class VoyageProvider(EmbeddingProvider):
    """Provider pour Voyage AI"""
    
    def __init__(self, api_key: str, model: str = "voyage-large-2"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.voyageai.com/v1/embeddings"

    def get_embedding(self, text: Union[str, List[str]]) -> Optional[List[float]]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "input": text if isinstance(text, list) else [text],
                "model": self.model
            }
            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"Erreur Voyage: {e}")
            return None

class CohereProvider(EmbeddingProvider):
    """Provider pour Cohere"""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.cohere.com/v2/embed"

    def get_embedding(self, text: Union[str, List[str]]) -> Optional[List[float]]:
        try:
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": self.model,
                "texts": text if isinstance(text, list) else [text],
                "input_type": "classification",
                "embedding_types": ["float"]
            }
            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["embeddings"]["float"][0]
        except Exception as e:
            print(f"Erreur Cohere: {e}")
            return None

class EmbeddingGenerator:
    """Classe principale pour la génération d'embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialise le générateur d'embeddings"""
        self.config = config
        self.provider = self._initialize_provider()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Créer le dossier de sortie
        Path(config['paths']['output_base']).mkdir(parents=True, exist_ok=True)

    def _initialize_provider(self) -> EmbeddingProvider:
        """Initialise le provider approprié selon la configuration"""
        provider_config = self.config['api']['provider']
        provider_name = provider_config['name']
        api_key = provider_config['key']
        model = provider_config['model']

        providers = {
            'openai': OpenAIProvider,
            'mistral': MistralProvider,
            'voyage': VoyageProvider,
            'cohere': CohereProvider
        }

        if provider_name not in providers:
            raise ValueError(f"Provider non supporté: {provider_name}")

        return providers[provider_name](api_key, model)

    def clean_text(self, text: str) -> str:
        """Nettoie le texte"""
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        """Divise le texte en chunks"""
        lines = text.split('\n')
        header_lines = self.config['processing']['header_lines']
        header = '\n'.join(lines[:header_lines]) + '\n'
        remaining_text = '\n'.join(lines[header_lines:])
        
        tokens = self.tokenizer.encode(remaining_text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = self.tokenizer.encode(header) + tokens[i:i + max_tokens]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
        
        return chunks

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Obtient l'embedding avec gestion des erreurs et retries"""
        max_retries = self.config['api']['max_retries']
        retry_delay = self.config['api']['retry_delay']
        
        for attempt in range(max_retries):
            try:
                return self.provider.get_embedding(text)
            except Exception as e:
                print(f"Tentative {attempt + 1}/{max_retries} échouée: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay ** attempt)
                else:
                    print(f"Échec après {max_retries} tentatives")
                    return None

    def process_file(self, file_path: str, chunk_size: int) -> List[Dict[str, Any]]:
        """Traite un fichier individuel"""
        filename = os.path.basename(file_path)
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            cleaned_content = self.clean_text(content)
            chunks = self.split_into_chunks(cleaned_content, chunk_size)
            
            print(f"Traitement de {filename}: {len(chunks)} chunks")
            
            for i, chunk in enumerate(tqdm(chunks, desc=f"Chunks de {filename}")):
                embedding = self.get_embedding(chunk)
                if embedding:
                    results.append({
                        'filename': filename,
                        'chunk_id': i,
                        'text': chunk,
                        'embedding': embedding
                    })
                
        except Exception as e:
            print(f"Erreur lors du traitement de {filename}: {e}")
            
        return results

    def save_results(self, results: List[Dict[str, Any]], chunk_size: int) -> None:
        """Sauvegarde les résultats dans différents formats"""
        output_base = Path(self.config['paths']['output_base'])
        folder_name = output_base / f"{chunk_size}tok"
        folder_name.mkdir(exist_ok=True)
        
        if 'csv' in self.config['output']['formats']:
            df = pd.DataFrame(results)
            df.to_csv(folder_name / f'embeddings_results_{chunk_size}tok.csv', index=False)
        
        if 'json' in self.config['output']['formats']:
            chunks = [{
                "text": r['text'],
                "embedding": r['embedding'],
                "metadata": {
                    "filename": r['filename'],
                    "chunk_id": r['chunk_id']
                }
            } for r in results]
            
            with open(folder_name / 'chunks.json', 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        if 'npy' in self.config['output']['formats']:
            embeddings = [r['embedding'] for r in results]
            embeddings_array = np.array(embeddings)
            np.save(folder_name / 'embeddings.npy', embeddings_array)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement de la configuration: {e}")

def main():
    """Fonction principale"""
    try:
        config = load_config()
        generator = EmbeddingGenerator(config)
        
        for chunk_size in config['processing']['chunk_sizes']:
            print(f"\nTraitement pour chunks de {chunk_size} tokens")
            all_results = []
            
            input_folder = config['paths']['input_folder']
            for filename in os.listdir(input_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(input_folder, filename)
                    results = generator.process_file(file_path, chunk_size)
                    all_results.extend(results)
            
            generator.save_results(all_results, chunk_size)
            print(f"Traitement terminé pour {chunk_size} tokens: {len(all_results)} chunks générés")
            
    except Exception as e:
        print(f"Erreur critique: {e}")

if __name__ == "__main__":
    main()

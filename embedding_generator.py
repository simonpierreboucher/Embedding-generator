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
import openai  # Assurez-vous que le module openai est importé

class EmbeddingProvider(ABC):
    """Classe abstraite pour les providers d'embeddings"""
    
    @abstractmethod
    def get_embedding(self, text: Union[str, List[str]]) -> Optional[List[float]]:
        """Obtient l'embedding pour un texte donné"""
        pass

# Vos classes de provider restent inchangées...

class EmbeddingGenerator:
    """Classe principale pour la génération d'embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialise le générateur d'embeddings"""
        self.config = config
        self.provider = self._initialize_provider()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configuration de l'API OpenAI pour LLM
        openai.api_key = self.config['api']['provider']['key']
        self.llm_model = self.config['api']['llm_model']
        self.llm_max_input_tokens = self.config['api']['llm_max_input_tokens']
        self.llm_max_output_tokens = self.config['api']['llm_max_output_tokens']
        self.max_retries = self.config['api']['max_retries']
        self.retry_delay = self.config['api']['retry_delay']
        
        # Créer le dossier de sortie
        Path(config['paths']['output_base']).mkdir(parents=True, exist_ok=True)

    # Votre méthode _initialize_provider reste inchangée

    # Méthodes clean_text et split_into_chunks restent inchangées
    
    def get_chunk_context_description(self, chunk_text: str, full_text: str) -> str:
        """Obtient une description du rôle du chunk dans le texte complet"""
        # Calcul des tokens nécessaires
        prompt_template = (
            "Voici un extrait (chunk) d'un texte :\n{chunk_text}\n\n"
            "Voici le texte complet ou partiel d'où provient l'extrait :\n{full_text}\n\n"
            "Pouvez-vous fournir une brève description du rôle de cet extrait dans le contexte du texte ?"
        )
        
        # Encode chunk_text and full_text
        chunk_tokens = self.tokenizer.encode(chunk_text)
        full_text_tokens = self.tokenizer.encode(full_text)
        prompt_tokens = self.tokenizer.encode(prompt_template.format(chunk_text="", full_text=""))

        total_tokens = len(chunk_tokens) + len(full_text_tokens) + len(prompt_tokens)
        
        if total_tokens > self.llm_max_input_tokens:
            # Tronquer le texte complet pour respecter la limite
            allowed_full_text_tokens = self.llm_max_input_tokens - len(chunk_tokens) - len(prompt_tokens)
            truncated_full_text_tokens = full_text_tokens[:allowed_full_text_tokens]
            truncated_full_text = self.tokenizer.decode(truncated_full_text_tokens)
        else:
            truncated_full_text = full_text
        
        # Préparer le prompt
        prompt = prompt_template.format(chunk_text=chunk_text, full_text=truncated_full_text)
        
        # Appel à l'API OpenAI pour obtenir la description
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "Vous êtes un assistant qui aide à résumer le rôle d'un extrait de texte dans le contexte du texte complet."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.llm_max_output_tokens,
                    temperature=0.7,
                    n=1,
                    stop=None
                )
                description = response['choices'][0]['message']['content'].strip()
                return description
            except Exception as e:
                print(f"Erreur lors de l'obtention de la description contextuelle (tentative {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** attempt)
                else:
                    print(f"Échec après {self.max_retries} tentatives")
                    return ""
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Obtient l'embedding avec gestion des erreurs et retries"""
        for attempt in range(self.max_retries):
            try:
                return self.provider.get_embedding(text)
            except Exception as e:
                print(f"Tentative {attempt + 1}/{self.max_retries} échouée: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay ** attempt)
                else:
                    print(f"Échec après {self.max_retries} tentatives")
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
                # Obtenir la description contextuelle
                description = self.get_chunk_context_description(chunk, cleaned_content)
                
                # Former le nouveau chunk
                new_chunk = f"{chunk}\n\nDescription du rôle dans le texte : {description}"
                
                # Obtenir l'embedding du nouveau chunk
                embedding = self.get_embedding(new_chunk)
                
                if embedding:
                    results.append({
                        'filename': filename,
                        'chunk_id': i,
                        'text': new_chunk,
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

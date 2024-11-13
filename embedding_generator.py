import os
import re
import yaml
import tiktoken
from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

class EmbeddingGenerator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le générateur d'embeddings avec la configuration.
        
        Args:
            config: Configuration chargée depuis le fichier YAML
        """
        self.config = config
        self.client = OpenAI(api_key=config['api']['key'])
        self.model = config['api']['model']
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Créer le dossier de sortie
        Path(config['paths']['output_base']).mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Nettoie le texte en supprimant les sauts de ligne et espaces multiples."""
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        """
        Divise le texte en chunks avec en-tête.
        
        Args:
            text: Texte à diviser
            max_tokens: Nombre maximum de tokens par chunk
        """
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
        """Obtient l'embedding d'un texte avec gestion des erreurs."""
        max_retries = self.config['api']['max_retries']
        retry_delay = self.config['api']['retry_delay']
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Tentative {attempt + 1}/{max_retries} échouée: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay ** attempt)
                else:
                    print(f"Échec après {max_retries} tentatives")
                    return None

    def process_file(self, file_path: str, chunk_size: int) -> List[Dict[str, Any]]:
        """Traite un fichier individuel."""
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
        """Sauvegarde les résultats dans les différents formats configurés."""
        output_base = Path(self.config['paths']['output_base'])
        folder_name = output_base / f"{chunk_size}tok"
        folder_name.mkdir(exist_ok=True)
        
        if 'csv' in self.config['output']['formats']:
            df = pd.DataFrame(results)
            df.to_csv(folder_name / f'embeddings_results_{chunk_size}tok.csv', index=False)
        
        if 'json' in self.config['output']['formats']:
            chunks = [{
                "text": r['text'],
                "embedding": r['chunk_id'],
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
    """Charge la configuration depuis le fichier YAML."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement de la configuration: {e}")

def main():
    try:
        # Charger la configuration
        config = load_config()
        
        # Initialiser le générateur
        generator = EmbeddingGenerator(config)
        
        # Traiter chaque taille de chunk
        for chunk_size in config['processing']['chunk_sizes']:
            print(f"\nTraitement pour chunks de {chunk_size} tokens")
            all_results = []
            
            # Traiter chaque fichier
            input_folder = config['paths']['input_folder']
            for filename in os.listdir(input_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(input_folder, filename)
                    results = generator.process_file(file_path, chunk_size)
                    all_results.extend(results)
            
            # Sauvegarder les résultats
            generator.save_results(all_results, chunk_size)
            print(f"Traitement terminé pour {chunk_size} tokens: {len(all_results)} chunks générés")
            
    except Exception as e:
        print(f"Erreur critique: {e}")

if __name__ == "__main__":
    main()

# TO DO: it looks like editing the subject (1) does not work properly.



#!/usr/bin/env python3
"""
Gronings Voorbeeldzinnen Generator
Een interactief spelletje om voorbeeldzinnen te maken voor Groningse werkwoordsvervoegingen.
"""

import pandas as pd
import random
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

class GroningsZinnenGenerator:
    def __init__(self, csv_path: str = "hogelandster_waarkwoorden.csv", 
                 save_path: str = "gronings_voorbeeldzinnen.json"):
        """
        Initialiseer de generator met een CSV bestand van werkwoorden.
        """
        self.csv_path = csv_path
        self.save_path = save_path
        self.df = pd.read_csv(csv_path)
        
        # Vervoegingskolommen (alles behalve index, vertaling en transitief)
        self.conjugation_columns = [col for col in self.df.columns 
                                   if col not in ['Vertaling', 'Transitief', 'Index']]
        
        # Onderwerpen per persoon/getal
        self.subjects = {
            '1sg': ['ik', 'k'],
            '2sg': ['doe', ''],
            '3sg': ['hai', 'zai', 'e', 't', 'dij', 'dij hond', 'dij man', 'dij vraauw', 
                   't jong', 't wicht', 'de boom', 'de auto', 'de fiets'],
            'Pl': ['wie', 'ie', 'joe', 'zai', 'dij', 'dij honden', 'dij mìnsen', 'de kinder', 'de bomen'],
            'Past-1sg': ['ik', 'k'],
            'Past-2sg': ['doe', ''],
            'Past-3sg': ['hai', 'zai', 'e', 't', 'dij', 'dij hond', 'dij man', 'dij vraauw', 
                   't jong', 't wicht', 'de boom', 'de auto', 'de fiets'],
            'Past-Pl': ['wie', 'ie', 'joe', 'zai', 'dij', 'dij honden', 'dij mìnsen', 'de kinder', 'de bomen'],
            'Part': ['ik heb', 'doe hest', 'hai het', 'zai het', 't het', 'dij het', 'wie hebben', 'ie hebben', 'joe hebben', 'zai hebben', 'dij hebben']
        }
        
        # Lijdend voorwerpen voor transitieve werkwoorden
        self.objects = [
            'de appel', 'de kouk', 'de kraant', 't bouk', 'de braif', 'de toavel',
            'mie', 'die', 'hom', 'heur', 't', 'e', 'dij', 'ons', 'joe', 'ze',
            'de deure', 'de toavel', 'de stoul', 't brood', 'de kovvie',
            'n bedie', 'wat', 'niks', 'ales', 'veul'
        ]
        
        # Extra zinsdelen (bijwoorden, voorzetsels, etc.)
        self.extras = [
            'haard', 'geern', 'mooi', 'goud', 'slim', 'vlot',
            'in hoes', 'op stroat', 'bie de boom', 'noar stad',
            'veur de eerste keer', 'mit plezaaier', 'zunder probleem',
            'elke dag', 'voak', 'noeit', 'aaltied', 'söms'
        ]
        
        # Laad of initialiseer de voorbeeldzinnen database
        self.load_or_init_database()
        
    def load_or_init_database(self):
        """Laad bestaande database of maak een nieuwe aan."""
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
            print(f"✓ Database geladen vanaf {self.save_path}")
        else:
            # Initialiseer lege database
            self.database = {}
            for idx, row in self.df.iterrows():
                infinitief = row['Infinitief']
                self.database[infinitief] = {}
                for col in self.conjugation_columns:
                    if pd.notna(row[col]) and row[col] != '':
                        self.database[infinitief][col] = {
                            'vorm': row[col],
                            'vertaling': row['Vertaling'],
                            'transitief': row['Transitief'],
                            'voorbeeldzin': None
                        }
            self.save_database()
            print(f"✓ Nieuwe database aangemaakt")
    
    def save_database(self):
        """Sla de database op naar schijf."""
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, ensure_ascii=False, indent=1)
    
    def get_stats(self) -> Dict:
        """Bereken statistieken over de voortgang."""
        total = 0
        filled = 0
        
        for werkwoord in self.database.values():
            for vervoeging in werkwoord.values():
                total += 1
                if vervoeging['voorbeeldzin'] is not None:
                    filled += 1
        
        percentage = (filled / total * 100) if total > 0 else 0
        return {
            'gedaan': filled,
            'te_gaan': total - filled,
            'totaal': total,
            'percentage': percentage
        }
    
    def get_random_empty_conjugation(self) -> Optional[Tuple[str, str, Dict]]:
        """
        Selecteer een willekeurige lege vervoeging.
        Retourneert (infinitief, vervoeging_type, data) of None als alles vol is.
        """
        empty_items = []
        
        for infinitief, werkwoord in self.database.items():
            for vervoeging_type, data in werkwoord.items():
                if data['voorbeeldzin'] is None:
                    empty_items.append((infinitief, vervoeging_type, data))
        
        if empty_items:
            return random.choice(empty_items)
        return None
    
    def suggest_sentence(self, infinitief: str, vervoeging_type: str, 
                        vorm: str, transitief: str) -> str:
        """Genereer een suggestie voor een voorbeeldzin."""
        # Kies passend onderwerp
        if vervoeging_type in self.subjects:
            subject = random.choice(self.subjects[vervoeging_type])
        else:
            subject = random.choice(self.subjects['3sg'])  # Default
        
        # Basis zin
        if vervoeging_type == 'Part':
            # Voltooid deelwoord heeft hulpwerkwoord nodig
            sentence = f"{subject} {vorm}"
        else:
            sentence = f"{subject} {vorm}"
        
        # Voeg object toe als transitief
        if transitief == 'ja':
            obj = random.choice(self.objects)
            sentence += f" {obj}"
        
        # Soms een extra toevoegen (30% kans)
        if random.random() < 0.3:
            extra = random.choice(self.extras)
            sentence += f" {extra}"
        
        return sentence + "."
    
    def edit_sentence_part(self, sentence: str, part_code: int, 
                          new_text: str) -> str:
        """
        Bewerk een deel van de zin.
        1 = onderwerp, 2 = lijdend voorwerp, 3 = meewerkend voorwerp, 4 = extra toevoegen
        """
        parts = sentence.rstrip('.').split()
        
        if part_code == 1:  # Onderwerp vervangen
            if len(parts) > 0:
                parts[0] = new_text
        elif part_code == 2:  # Lijdend voorwerp vervangen
            # Simpele benadering: vervang laatste substantief
            if len(parts) > 2:
                parts[-1] = new_text
        elif part_code == 4:  # Extra toevoegen
            parts.append(new_text)
        
        return ' '.join(parts) + '.'
    
    def play_game(self):
        """Start het interactieve spel."""
        print("\n" + "="*60)
        print("🎮 GRONINGS VOORBEELDZINNEN GENERATOR")
        print("="*60)
        
        while True:
            # Toon statistieken
            stats = self.get_stats()
            print(f"\n📊 Voortgang: {stats['gedaan']}/{stats['totaal']} " +
                  f"({stats['percentage']:.1f}%) - Nog te gaan: {stats['te_gaan']}")
            print("-"*60)
            
            # Haal random lege vervoeging
            result = self.get_random_empty_conjugation()
            
            if result is None:
                print("\n🎉 GEFELICITEERD! Alle vervoegingen hebben een voorbeeldzin!")
                break
            
            infinitief, vervoeging_type, data = result
            vorm = data['vorm']
            vertaling = data['vertaling']
            transitief = data['transitief']
            
            print(f"\n📝 Werkwoord: {infinitief}")
            print(f"   Nederlandse vertaling: {vertaling}")
            print(f"   Vervoeging: {vervoeging_type}")
            print(f"   Vorm: {vorm}")
            print(f"   Transitief: {transitief}")
            
            # Genereer suggestie
            suggestion = self.suggest_sentence(infinitief, vervoeging_type, 
                                              vorm, transitief)
            print(f"\n💡 Suggestie: {suggestion}")
            
            current_sentence = suggestion
            
            while True:
                print("\n[A]ccepteren | [1-4] Bewerk deel | [H]andmatig | [S]kip | [Q]uit")
                choice = input("Keuze: ").strip().upper()
                
                if choice == 'A':
                    # Accepteer de zin
                    self.database[infinitief][vervoeging_type]['voorbeeldzin'] = current_sentence
                    self.save_database()
                    print("✓ Zin geaccepteerd en opgeslagen!")
                    break
                    
                elif choice in ['1', '2', '3', '4']:
                    # Bewerk deel van de zin
                    part_names = {
                        '1': 'onderwerp',
                        '2': 'lijdend voorwerp',
                        '3': 'meewerkend voorwerp',
                        '4': 'extra toevoegen'
                    }
                    print(f"\nNieuwe tekst voor {part_names[choice]}:")
                    new_text = input("> ").strip()
                    if new_text:
                        current_sentence = self.edit_sentence_part(
                            current_sentence, int(choice), new_text
                        )
                        print(f"📝 Nieuwe zin: {current_sentence}")
                
                elif choice == 'H':
                    # Handmatig hele zin invoeren
                    print("\nVoer de complete zin in:")
                    manual_sentence = input("> ").strip()
                    if manual_sentence:
                        if not manual_sentence.endswith('.'):
                            manual_sentence += '.'
                        current_sentence = manual_sentence
                        print(f"📝 Nieuwe zin: {current_sentence}")
                
                elif choice == 'S':
                    # Skip deze vervoeging
                    print("⏭️  Overgeslagen")
                    break
                
                elif choice == 'Q':
                    # Stop het spel
                    print("\n👋 Tot ziens! Je voortgang is opgeslagen.")
                    return
                
                else:
                    print("❌ Ongeldige keuze")

def main():
    """Hoofdfunctie."""
    print("🚀 Gronings Voorbeeldzinnen Generator wordt gestart...")
    
    # Check of CSV bestaat
    csv_path = "hogelandster_waarkwoorden.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Fout: {csv_path} niet gevonden!")
        print("Zorg dat het CSV bestand in dezelfde map staat als dit script.")
        sys.exit(1)
    
    # Start de generator
    generator = GroningsZinnenGenerator(csv_path)
    
    try:
        generator.play_game()
    except KeyboardInterrupt:
        print("\n\n👋 Programma onderbroken. Je voortgang is opgeslagen!")
    except Exception as e:
        print(f"\n❌ Er is een fout opgetreden: {e}")
        print("Je voortgang tot nu toe is opgeslagen.")

if __name__ == "__main__":
    main()
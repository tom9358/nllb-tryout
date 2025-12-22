# TO DO: try out more to see if it really works as intended
# the parentheses stuff seems not in the right places e.g.💡 Suggestie: dij|snieden (snidt ⊲ sniedt)|hom|in hoes.


#!/usr/bin/env python3
"""
Gronings Voorbeeldzinnen Generator
Een interactief spelletje om voorbeeldzinnen te maken voor Groningse werkwoordsvervoegingen.
Met optionele LLM-ondersteuning voor betere zinnen.
"""

import pandas as pd
import random
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# Optionele imports voor LLM ondersteuning
try:
    import torch
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  Transformers/torch niet geïnstalleerd. LLM features uitgeschakeld.")
    print("   Installeer met: pip install transformers torch")

class GroningsZinnenGenerator:
    def __init__(self, csv_path: str = "hogelandster_waarkwoorden.csv", 
                 save_path: str = "gronings_voorbeeldzinnen.json",
                 use_llm: bool = False):
        """
        Initialiseer de generator met een CSV bestand van werkwoorden.
        
        Args:
            csv_path: Pad naar het CSV bestand
            save_path: Pad voor de JSON database
            use_llm: Gebruik LLM voor betere zinssuggesties (vereist GPU)
        """
        self.csv_path = csv_path
        self.save_path = save_path
        self.use_llm = use_llm and LLM_AVAILABLE
        self.df = pd.read_csv(csv_path)

        # Vervoegingskolommen (alles behalve index, vertaling en transitief)
        self.conjugation_columns = [col for col in self.df.columns 
                                   if col not in ['Vertaling', 'Transitief']]

        # Initialiseer LLM indien gewenst
        if self.use_llm:
            self.init_llm()

        # Onderwerpen per persoon/getal
        self.subjects = {
            'Infinitief': ['k bin aan t', 'most', 'zai kin', 'hai kin', 'wie zellen', 
                          'ie binnen aan t', 'joe goan', 't is van belang om te',
                          'doe zugst mie', 'hai wil', 't is tied om te'],
            '1sg': ['ik', 'k'],
            '2sg': ['doe', ''],
            '3sg': ['hai', 'zai', 'e', 't', 'dij', 'dizze hond', 'dij kat', 'dij man',
                   'dij vraauw', 't jong', 't wicht', 'de boom', 'de auto', 'de fiets', 'dien opoe',
                   'mien noaber', 'de bakker', 'juffraauw', 'mien oma', 'heur opa', 'dien zuske', 'zien bruier'],
            'Pl': ['wie', 'ie', 'joe', 'zai', 'dij', 'dij honden', 'dij mìnsen',
                  'de kinder', 'de bomen', 'dien noabers', 'de vrunden', 'mien olders'],
            'Past-1sg': ['ik', 'k'],
            'Past-2sg': ['doe', ''],
            'Past-3sg': ['hai', 'zai', 'e', 't', 'dij', 'dizze hond', 'dij kat', 'dij man',
                   'dij vraauw', 't jong', 't wicht', 'de boom', 'de auto', 'de fiets', 'dien opoe',
                   'mien noaber', 'de bakker', 'juffraauw', 'mien oma', 'heur opa', 'dien zuske', 'zien bruier'],
            'Past-Pl': ['wie', 'ie', 'joe', 'zai', 'dij', 'dij honden', 'dij mìnsen',
                  'de kinder', 'de bomen', 'dien noabers', 'de vrunden', 'mien olders'],
            'Part': ['ik heb', 'doe hest', 'hai het', 'zai het', 't het', 'dij het', 
                    'wie hebben', 'ie hebben', 'joe hebben', 'zai hebben', 'dij hebben',
                    'mien noaber het', 'de bakker het', 'de kinder hebben', 'mien olders hebben', 'dij lu hebben'],
        }

        # Lijdend voorwerpen voor transitieve werkwoorden
        self.objects = [
            'de appel', 'de kouk', 'de kraant', 't bouk', 'de braif', 'de toavel',
            'mie', 'die', 'hom', 'heur', 't', 'e', 'dij', 'ons', 'joe', 'ze',
            'de deure', 'de stoul', 't brood', 'de kovvie', 't glaas', 'n tazze',
            't waark', 'n bosschop', 'de hond', 't eten', 'de waske', 'de auto', 't huus',
            'n verhoal', 'de woarhaid', 'n laidje', 't geld', 'sleudels', 'lewaai'
        ]

        # Extra zinsdelen
        self.extras = [
            'n bedie', 'wat', 'niks', 'ales', 'veul', 'waaineg',
            'haard', 'geern', 'mooi', 'goud', 'slim', 'vlot', 'rusteg',
            'in hoes', 'op stroat', 'bie de boom', 'noar stad', 'in t dörp',
            'veur de eerste keer', 'mit plezaaier', 'zunder probleem',
            'elke dag', 'voak', 'noeit', 'aaltied', 'söms', 'n moal',
            'vanmörn', 'guster', 'straks', 'makkelk', 'in toene',
            'op zundagmörn'
        ]

        # Werkwoord-specifieke contexten voor natuurlijkere zinnen
        self.verb_contexts = {
            'eten': ['t brood', 'de soep', 'n appel', 'de paardeblomen'],
            'lopen': ['noar hoes', 'over stroat', 'in t park', 'haard'],
            'slapen': ['in bèrre', 'lekker', 'laang', 'as n roos'],
            'werken': ['haard', 'in de schure', 'op t laand', 'bie de gemeente'],
            'zeggen': ['de woarhaid', 'wat', 'niks', 'dat t nait gait'],
            'maken': ['t eten', 'n plan', 'kabaal', 'n tekenen'],
            'gaan': ['noar hoes', 'vot', 'noar bèrre', 'mit vakantie'],
            'komen': ['thoes', 'laater', 'mörn', 'op tied'],
            'blijven': ['thoes', 'hier', 'zitten', 'woar e is'],
            'schrijven': ['n braif', 'n verhoal', 'mooi', 'mit potlood']
        }

        # Laad of initialiseer de voorbeeldzinnen database
        self.load_or_init_database()

    def init_llm(self):
        """Initialiseer het LLM model voor betere zinssuggesties."""
        try:
            print("🤖 LLM model wordt geladen...")
            # We gebruiken een klein Nederlands model dat ook Gronings begrijpt
            model_name = "GroNLP/gpt2-small-dutch"  # Of "yhavinga/gpt2-medium-dutch"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Gebruik GPU indien beschikbaar
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✓ LLM geladen op GPU")
            else:
                print("✓ LLM geladen op CPU (langzamer)")
                
            # Stel padding token in
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            print(f"⚠️  LLM kon niet geladen worden: {e}")
            self.use_llm = False

    def process_verb_form(self, vorm: str) -> Tuple[str, Optional[str]]:
        """
        Verwerk werkwoordsvormen met haakjes volgens de regels:
        - Enkele letter zonder spatie ervoor: haakjes weg, letter behouden (mat(t)en -> matten)
        - Alle andere gevallen: tekst tussen haakjes verwijderen
        
        Returns:
            (verwerkte_vorm, alternatieve_vorm)
        """
        if not vorm or pd.isna(vorm):
            return vorm, None
        
        original = vorm
        alternative = None
        
        # Patroon voor enkele letter direct na tekst (geen spatie)
        single_char_pattern = r'([^\s])\((.)\)'
        
        # Check voor single character pattern
        if re.search(single_char_pattern, vorm):
            # Vervang door letter zonder haakjes
            vorm = re.sub(single_char_pattern, r'\1\2', vorm)
        
        # Nu verwijder alle overige tekst tussen haakjes
        if '(' in vorm:
            # Bewaar de alternatieve vorm
            alternative_match = re.search(r'\((.*?)\)', vorm)
            if alternative_match:
                alternative = alternative_match.group(1).strip()
            # Verwijder alles tussen haakjes
            vorm = re.sub(r'\s*\([^)]*\)', '', vorm).strip()
        
        # Als er meerdere opties zijn gescheiden door ⊲, neem de eerste
        if '⊲' in vorm:
            parts = vorm.split('⊲')
            vorm = parts[0].strip()
            if len(parts) > 1 and not alternative:
                alternative = parts[1].strip()
        
        return vorm, alternative

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
                        # Verwerk de vorm met haakjes
                        vorm, alternatief = self.process_verb_form(str(row[col]))
                        
                        self.database[infinitief][col] = {
                            'vorm': vorm,
                            'vorm_origineel': str(row[col]),  # Bewaar origineel
                            'alternatief': alternatief,  # Bewaar alternatieve vorm
                            'vertaling': row['Vertaling'],
                            'transitief': row['Transitief'],
                            'voorbeeldzin': None
                        }
            self.save_database()
            print(f"✓ Nieuwe database aangemaakt")

    def save_database(self):
        """Sla de database op naar schijf."""
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, ensure_ascii=False, indent=2)

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
    
    def get_contextual_object(self, vertaling: str, transitief: str) -> str:
        """Kies een contextueel passend object voor het werkwoord."""
        if transitief != 'ja':
            return ""
        
        # Check of we specifieke context hebben voor dit werkwoord
        for verb_key, contexts in self.verb_contexts.items():
            if verb_key in vertaling.lower():
                return random.choice(contexts)
        
        # Anders gebruik algemene objecten
        return random.choice(self.objects)
    
    def get_contextual_extra(self, vertaling: str) -> str:
        """Kies een contextueel passend extra zinsdeel."""
        # Check werkwoord-specifieke contexten
        for verb_key, contexts in self.verb_contexts.items():
            if verb_key in vertaling.lower():
                # Voor sommige werkwoorden zijn de contexten ook goede extras
                if random.random() < 0.3:  # 30% kans
                    return random.choice(contexts)
        
        # Gebruik algemene extras
        if random.random() < 0.3:  # 30% kans
            return random.choice(self.extras)
        return ""
    
    def suggest_sentence_llm(self, infinitief: str, vervoeging_type: str,
                            vorm: str, transitief: str, vertaling: str) -> str:
        """Genereer een zin met behulp van een LLM."""
        if not self.use_llm:
            return self.suggest_sentence_simple(infinitief, vervoeging_type, 
                                               vorm, transitief, vertaling)
        
        try:
            # Maak een prompt voor het model
            subject = random.choice(self.subjects.get(vervoeging_type, self.subjects['3sg']))
            
            prompt = f"""Maak een natuurlijke Groningse zin met:
Onderwerp: {subject}
Werkwoord: {vorm} (van {infinitief}, Nederlands: {vertaling})
Transitief: {transitief}

Groningse zin:"""
            
            # Genereer met het model
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 30,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sentence = generated.split("Groningse zin:")[-1].strip()
            
            # Valideer en format de zin
            if vorm in sentence and subject in sentence:
                return sentence if sentence.endswith('.') else sentence + '.'
            else:
                # Fallback naar simpele methode
                return self.suggest_sentence_simple(infinitief, vervoeging_type,
                                                   vorm, transitief, vertaling)
                
        except Exception as e:
            print(f"⚠️  LLM generatie mislukt: {e}")
            return self.suggest_sentence_simple(infinitief, vervoeging_type,
                                               vorm, transitief, vertaling)
    
    def suggest_sentence_simple(self, infinitief: str, vervoeging_type: str, 
                               vorm: str, transitief: str, vertaling: str) -> str:
        """Genereer een simpele suggestie voor een voorbeeldzin met |-markers."""
        # Kies passend onderwerp
        try:
            subject = random.choice(self.subjects[vervoeging_type])
        except KeyError:
            subject = random.choice(self.subjects['3sg'])  # Fallback
        
        # Kies contextueel passend object en extra
        obj = self.get_contextual_object(vertaling, transitief) if transitief == 'ja' else ""
        extra = self.get_contextual_extra(vertaling)

        # Bouw de zin op basis van het type vervoeging
        if vervoeging_type in ['Part', 'Infinitief']:
            # Niet-persoonsvormen komen achteraan
            parts = [subject]
            if obj:
                parts.append(obj)
            if extra:
                parts.append(extra)
            parts.append(vorm)
        else:
            # Persoonsvormen komen direct na het onderwerp
            parts = [subject, vorm]
            if obj:
                parts.append(obj)
            if extra:
                parts.append(extra)

        # Filter lege delen
        parts = [p for p in parts if p]
        return '|'.join(parts) + '.'
    
    def suggest_sentence(self, infinitief: str, vervoeging_type: str,
                        vorm: str, transitief: str, vertaling: str) -> str:
        """Hoofdmethode voor zinssuggesties."""
        if self.use_llm:
            return self.suggest_sentence_llm(infinitief, vervoeging_type,
                                            vorm, transitief, vertaling)
        else:
            return self.suggest_sentence_simple(infinitief, vervoeging_type,
                                               vorm, transitief, vertaling)

    def edit_sentence_part(self, parts: list[str], part_code: int, 
                          new_text: str) -> str:
        """
        Bewerk een deel van de zin.
        Gebruik hele getallen vanaf 0 om het zinsdeel te specificeren.
        """
        if part_code < len(parts):
            if new_text:
                parts[part_code] = new_text
            else:
                # Verwijder het deel als nieuwe tekst leeg is
                parts.pop(part_code)
        else:
            # Voeg toe als het deel nog niet bestaat
            if new_text:
                parts.append(new_text)

        # Filter lege delen en rebuild
        parts = [p for p in parts if p]
        return '|'.join(parts) + '.'

    def play_game(self):
        """Start het interactieve spel."""
        print("\n" + "="*60)
        print("🎮 GRONINGS VOORBEELDZINNEN GENERATOR")
        if self.use_llm:
            print("🤖 LLM-ondersteuning actief")
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
            vorm_origineel = data.get('vorm_origineel', vorm)
            alternatief = data.get('alternatief')
            vertaling = data['vertaling']
            transitief = data['transitief']
            
            print(f"\n📝 Werkwoord: {infinitief}")
            print(f"   Nederlandse vertaling: {vertaling}")
            print(f"   Vervoeging: {vervoeging_type}")
            print(f"   Vorm: {vorm}")
            if vorm != vorm_origineel:
                print(f"   (Origineel: {vorm_origineel})")
            if alternatief:
                print(f"   (Alternatief: {alternatief})")
            print(f"   Transitief: {transitief}")
            
            # Genereer suggestie
            suggestion = self.suggest_sentence(infinitief, vervoeging_type, 
                                              vorm, transitief, vertaling)
            print(f"\n💡 Suggestie: {suggestion}")
            
            current_sentence = suggestion
            
            while True:
                print("\n[A]ccepteren | [0-9] Bewerk deel | [N]ieuwe suggestie | [H]andmatig | [S]kip | [Q]uit")
                choice = input("Keuze: ").strip().upper()
                
                if choice == 'A':
                    # Accepteer de zin
                    self.database[infinitief][vervoeging_type]['voorbeeldzin'] = current_sentence
                    self.save_database()
                    print("✓ Zin geaccepteerd en opgeslagen!")
                    break
                
                elif choice == 'N':
                    # Genereer nieuwe suggestie
                    suggestion = self.suggest_sentence(infinitief, vervoeging_type,
                                                      vorm, transitief, vertaling)
                    current_sentence = suggestion
                    print(f"💡 Nieuwe suggestie: {suggestion}")
                    
                elif choice.isdigit():
                    # Bewerk deel van de zin
                    part_num = int(choice)
                    # Verwijder punt en splits op |
                    core = current_sentence.rstrip('.')
                    parts = core.split('|')
                    
                    if part_num < len(parts):
                        print(f"\nNieuwe tekst voor '{parts[part_num]}' (leeg = verwijderen):")
                    else:
                        print(f"\nNieuw deel toevoegen op positie {part_num}:")
                    
                    new_text = input("> ").strip()
                    current_sentence = self.edit_sentence_part(parts, part_num, new_text)
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
    
    # Vraag of LLM gebruikt moet worden
    use_llm = False
    if LLM_AVAILABLE:
        print("\n🤖 Wil je LLM-ondersteuning gebruiken voor betere zinssuggesties?")
        print("   (Dit vereist een GPU en kan de eerste keer lang duren)")
        response = input("   [j/n]: ").strip().lower()
        use_llm = response == 'j'
    
    # Start de generator
    generator = GroningsZinnenGenerator(csv_path, use_llm=use_llm)
    
    try:
        generator.play_game()
    except KeyboardInterrupt:
        print("\n\n👋 Programma onderbroken. Je voortgang is opgeslagen!")
    except Exception as e:
        print(f"\n❌ Er is een fout opgetreden: {e}")
        import traceback
        traceback.print_exc()
        print("Je voortgang tot nu toe is opgeslagen.")

if __name__ == "__main__":
    main()
import os
import shutil

def raggruppa_per_nome(strings):
    nome_dict={}

    for string in strings:
        nome, numero=string.rsplit('_',1)
        if nome in nome_dict:
            nome_dict[nome].append(string)
        else:
            nome_dict[nome]=[string]
    
    raggruppate=list(nome_dict.values())

    for lista in raggruppate:
        lista.sort(key=lambda x: int(x.split('_')[-1]))

    return raggruppate

def split_name_and_number(string):
    parts=string.rsplit('_',1)
    number=parts[-1]
    name="_".join(parts[:-1])
    return name,number

def crea_cartelle_e_copia_contenuto(risultato,cartella_input, cartella_output):
    if(len(risultato) < 2):
        #print(f"QUI CE NE SONO MENO DI 2: {risultato}")
        return 
    
    for i in range(len(risultato)-1):
        nome_cartella_output=risultato[i]
        percorso_cartella_output=os.path.join(cartella_output,nome_cartella_output)
        os.makedirs(percorso_cartella_output,exist_ok=True)

        #print(lista_cartelle)
        
        number=1
        for cartella in risultato[i:i+2]:
            #print(cartella)
            origine=os.path.join(cartella_input,cartella)
            for file in os.listdir(origine):
                destinazione=os.path.join(percorso_cartella_output,f'{number}.jpg')
                print(f"Questo e\' il nome del file: {file} proveniente da {origine} e va in {destinazione}")
                shutil.copy2(os.path.join(origine,file),destinazione)
                number+=1

def folder(input_folder, output_folder):
    #if the folder doesn't exist, i create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder_name in os.listdir(input_folder):
        #print(folder_name)
        folder_path=os.path.join(input_folder,folder_name)
        #print(folder_path)
        if not os.path.isdir(folder_path):
            continue

        #I create in the output folder a folder that has the same name as folder_name
        output_subfolder=os.path.join(output_folder,folder_name)
        print(output_subfolder)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        files=os.listdir(folder_path)
        #print(file)
        files=raggruppa_per_nome(files)

        #print(files)
        #print(f"print del folder_path: {folder_path}")
        for file in files:
            #print(file)
            crea_cartelle_e_copia_contenuto(file,folder_path,output_subfolder)

input_folder='Flow_HMDB51_Dataset'
output_folder_frames='Flow_HMDB51_2'

folder(input_folder,output_folder_frames)
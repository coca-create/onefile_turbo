import gradio as gr
import docx
from docx import Document
import tempfile
import os
import re
import zipfile
from tab4 import tab4_func as t4
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from datetime import datetime

def webvtt_remover_NR(sentence):
    sentence = re.sub(r'[\u200B-\u200D\uFEFF]', '', sentence)
    sentence = sentence.replace('\r\n', '\n').replace('\r', '\n')
    sentence = sentence.replace('\u200B',"")

    pattern1 = re.compile(r'(\d+)(\d{1}:\d{2}:\d{2}\.\d{3}-->\d{1}:\d{2}:\d{2}\.\d{3})', re.DOTALL)
    pattern2 = re.compile(r'(\d+)(\d{2}:\d{2}:\d{2}\.\d{3}-->\d{2}:\d{2}:\d{2}\.\d{3})', re.DOTALL)

    match = pattern1.search(sentence)
    #print(f"webvtt_remover_match[search]_pattern1(d1):{match}")
    if match:
        start_index = match.start(1)
        rm_webvtt_sentence = sentence[start_index:]
    else:
        match = pattern2.search(sentence)
        #print(f"webvtt_remover_match[search]_pattern2(d2):{match}")
        if match:
            start_index = match.start(1)
            rm_webvtt_sentence = sentence[start_index:]
        else:
            rm_webvtt_sentence = sentence

    return rm_webvtt_sentence
'''def webvtt_remover_NR(sentence):
    sentence = re.sub(r'[\u200B-\u200D\uFEFF]', '', sentence)
    sentence = sentence.replace('\r\n', '\n').replace('\r', '\n')
    sentence = sentence.replace('\u200B',"")

    pattern = re.compile(r'(\d+)(\d{1}:\d{2}:\d{2}\.\d{3}-->\d{1}:\d{2}:\d{2}\.\d{3})', re.DOTALL)
    match = pattern.search(sentence)
    if match:
        start_index = match.start(1)
        rm_webvtt_sentence = sentence[start_index:]
    elif pattern.search(re.compile(r'(\d+)(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3})', re.DOTALL)
        ):
        match =pattern.search(re.compile(r'(\d+)(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3})', re.DOTALL)
        )
        start_index = match.start(1)
        rm_webvtt_sentence = sentence[start_index]  
    else:
        rm_webvtt_sentence = sentence
    return rm_webvtt_sentence'''
def remove_special_characters(text):
    # 特殊文字を除去する関数
    return re.sub(r'[\u200B-\u200D\uFEFF]', '', text).replace('\u200B', '')


def clean_excel_data(df):
    # DataFrameのすべてのセルの特殊文字を除去する関数
    for col in df.columns:
        df[col] = df[col].apply(lambda x: remove_special_characters(str(x)))
    return df

def excel_to_srt(excel_path, output_srt_path):
    # Excelファイルを読み込み
    df = pd.read_excel(excel_path)
    # 特殊文字を除去
    df = clean_excel_data(df)
    # SRTファイルに書き込み
    with open(output_srt_path, 'w', encoding='utf-8') as srt_file:
        for index, row in df.iterrows():
            srt_file.write(f"{(int(row.iloc[0]))}\n")
            srt_file.write(f"{(str(row.iloc[1]).replace('.', ',')).strip()} --> {(str(row.iloc[2]).replace('.', ',')).strip()}\n")
            srt_file.write(f"{(row.iloc[3]).strip()}\n\n")
    return output_srt_path

def excel_to_vtt(excel_path, output_vtt_path):
    # Excelファイルを読み込み
    df = pd.read_excel(excel_path)
    # 特殊文字を除去
    df = clean_excel_data(df)
    # VTTファイルに書き込み
    with open(output_vtt_path, 'w', encoding='utf-8') as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for index, row in df.iterrows():
            vtt_file.write(f"{(int(row.iloc[0]))}\n")
            vtt_file.write(f"{(str(row.iloc[1]).replace(',', '.')).strip()} --> {(str(row.iloc[2]).replace(',', '.')).strip()}\n")
            vtt_file.write(f"{(row.iloc[3]).strip()}\n\n")
    return output_vtt_path

def convert_docx_to_srttxt(docx_files):
    output_files = []
    if docx_files is None:
        return []
    for docx_file in docx_files:
        try:
            filename = os.path.basename(docx_file)
            base_name, ext = os.path.splitext(filename)
            clean_name = re.sub(r'[\r\n]+', '', base_name)
            #print(f"Processing file: {clean_name}")

            if clean_name.endswith('_srt 1'):
                output_filename = clean_name.replace('_srt 1', '_ja.srt')
            elif clean_name.endswith("_srt"):
                output_filename = clean_name.replace("_srt", "_ja.srt")
            elif clean_name.endswith('_vtt 1'):
                output_filename = clean_name.replace('_vtt 1', '_ja.vtt')
            elif clean_name.endswith("_vtt"):
                output_filename = clean_name.replace("_vtt", "_ja.vtt")
            elif clean_name.endswith('_txtnr 1'):
                output_filename = clean_name.replace('_txtnr 1', '_NR_ja.txt')
            elif clean_name.endswith("_txtnr"):
                output_filename = clean_name.replace("_txtnr", "_NR_ja.txt")
            elif clean_name.endswith('_txtr 1'):
                output_filename = clean_name.replace('_txtr 1', '_R_ja.txt')
            elif clean_name.endswith("_txtr"):
                output_filename = clean_name.replace("_txtr", "_R_ja.txt")
            else:
                print(f"Skipping file with unrecognized pattern: {clean_name}")
                continue
            timestamp_patch = datetime.now().strftime("%Y%m%d%H%M%S")
            temp_dir = os.path.join(tempfile.gettempdir(), f"tempdir_{timestamp_patch}")
            os.makedirs(temp_dir, exist_ok=True)
            output_filepath = os.path.join(temp_dir, output_filename)

            if output_filename.endswith('.txt'):
                doc = Document(docx_file)
                content = "\n".join([para.text for para in doc.paragraphs])
                #print(f"Initial content read from file: {content[:100]}")  # Show only the first 100 characters for brevity

            if output_filename.endswith('.srt'):
                output_filepath=excel_to_srt(docx_file,output_filepath)

            elif output_filename.endswith('vtt'):
                output_filepath=excel_to_vtt(docx_file,output_filepath)

            
            if output_filename.endswith('.txt'):
                final_content = content

                with open(output_filepath, 'w', encoding='utf-8') as output_file:
                    output_file.write(final_content)
            
            output_files.append(output_filepath)
        except Exception as e:
            print(f"An error occurred while processing {filename}: {str(e)}")

    if len(output_files) > 1:
        timestamp_patch = datetime.now().strftime("%Y%m%d%H%M%S")
        temp_dir = os.path.join(tempfile.gettempdir(), f"tempdir_{timestamp_patch}")
        os.makedirs(temp_dir, exist_ok=True)
        zip_filename = os.path.join(temp_dir, "converted_from_docx_ja.zip")
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in output_files:
                zipf.write(file, os.path.basename(file))
        
        output_files.append(zip_filename)
    
    return output_files

def clear_inputs():
    return None, None



def process_doc_files(files):
    output_files = []
    if files is None:
        return []
    for file in files:
        filename = os.path.basename(file)
        #print(filename)
        match = re.match(r"(.+?)(_NR\.txt|_R\.txt|\.srt|\.vtt)$", filename)
        if not match:
            continue  # skip files with unknown extensions
        
        basename, ext = match.groups()
        if ext == ".srt":
            doc_filename = f"{basename}_srt.xlsx"
        elif ext == ".vtt":
            doc_filename = f"{basename}_vtt.xlsx"
        elif ext == "_NR.txt":
            doc_filename = f"{basename}_txtnr.docx"
        elif ext == "_R.txt":
            doc_filename = f"{basename}_txtr.docx"
        timestamp_patch = datetime.now().strftime("%Y%m%d%H%M%S")
        temp_dir = os.path.join(tempfile.gettempdir(), f"tempdir_{timestamp_patch}")
        os.makedirs(temp_dir, exist_ok=True)

        doc_filename=os.path.join(temp_dir,doc_filename)

        if ext in ['.srt', '.vtt']:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            #print(content)
            
            if ext == '.srt':
                unified_content = t4.unify_timestamps(content)
                content = parse_srt_c(unified_content)
            else:
                unified_content = t4.unify_timestamps_vtt(content)
                content = parse_vtt_c(unified_content)

            t5_excel_path, t5_df = create_excel_from_srt_c(content, doc_filename)
            output_files.append(t5_excel_path)
            #print(f"Processed file: {t5_excel_path}")
        
        elif ext in ["_NR.txt", "_R.txt"]:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            doc = Document()
            doc.add_paragraph(content)
            doc.save(doc_filename)
            output_files.append(doc_filename)
    
    if len(output_files) > 1:

        zip_filename = "converted_from_srttxt_en.zip"
        zip_filename = os.path.join(temp_dir,zip_filename)
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            for file in output_files:
                zip_file.write(file, os.path.basename(file))
        output_files.append(zip_filename)
    
    return output_files

def clear_both():
    return None, None


#つじつま合わせの追加。
def parse_srt_c(srt_content):
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?:\n\n|\Z)'
    matches = re.findall(pattern, srt_content, re.DOTALL)
    if not matches:
        pattern = r'(\d+)\n(\d{1}:\d{2}:\d{2}\,\d{3}) --> (\d{1}:\d{2}:\d{2}\,\d{3})\n(.*?)(?:\n\n|\Z)'
        matches = re.findall(pattern, srt_content, re.DOTALL)    
    subtitles = []
    for match in matches:
        subtitles.append({
            'ID': int(match[0]),
            'Start': match[1],
            'End': match[2],
            'Text': match[3].replace('\n', ' ')
        })
    
    return subtitles

def parse_vtt_c(vtt_content):
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.*?)(?:\n\n|\Z)'
    matches = re.findall(pattern, vtt_content, re.DOTALL)
    if not matches:
        pattern = r'(\d+)\n(\d{1}:\d{2}:\d{2}\.\d{3}) --> (\d{1}:\d{2}:\d{2}\.\d{3})\n(.*?)(?:\n\n|\Z)'
        matches = re.findall(pattern, vtt_content, re.DOTALL)
    
    subtitles = []
    for match in matches:
        subtitles.append({
            'ID': int(match[0]),
            'Start': match[1],
            'End': match[2],
            'Text': match[3].replace('\n', ' ')
        })
    
    return subtitles

def create_excel_from_srt_c(srt_content, file_name):
    english_subtitles = srt_content

    data = []
    for eng in english_subtitles:
        data.append({
            'ID': eng['ID'],
            'Start': eng['Start'],
            'End': eng['End'],
            'English Subtitle': eng['Text']
        })

    df = pd.DataFrame(data)
    timestamp_patch = datetime.now().strftime("%Y%m%d%H%M%S")
    temp_dir = os.path.join(tempfile.gettempdir(), f"tempdir_{timestamp_patch}")
    os.makedirs(temp_dir, exist_ok=True)

    excel_file_path = os.path.join(temp_dir, file_name)
    
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Subtitles')
        workbook = writer.book
        worksheet = writer.sheets['Subtitles']

        column_widths = {'A': 7, 'B': 25, 'C': 25, 'D': 90, 'E': 90}
        for column, width in column_widths.items():
            worksheet.column_dimensions[column].width = width

        for row in worksheet.iter_rows(min_row=2, max_row=len(df) + 1):
            for cell in row:
                if cell.column_letter == 'A':
                    cell.alignment = Alignment(horizontal='right', vertical='center')
                elif cell.column_letter in ['B', 'C']:
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                elif cell.column_letter in ['D', 'E']:
                    cell.alignment = Alignment(horizontal='left', vertical='center')

        for row in worksheet.iter_rows(min_row=2, max_row=len(df) + 1):
            worksheet.row_dimensions[row[0].row].height = 30

        header_font = Font(bold=True)
        for cell in worksheet["1:1"]:
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')
            cell.fill = PatternFill(start_color="DAEEF3", end_color="DAEEF3", fill_type="solid")
    
    return excel_file_path, df
import gradio as gr
import re
import os
from datetime import timedelta
import srt
import zipfile
import tempfile
from datetime import datetime
from tab7 import tab7_func as t7

def parse_vtt_time(time_str):
    match = re.match(r"(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    groups = match.groups()
    hours = int(groups[0] or 0)
    minutes = int(groups[1])
    seconds = int(groups[2])
    milliseconds = int(groups[3])
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

def format_vtt_time(delta):
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def split_vtt_segment(segment, current_id):
    text = segment['text'].replace('。」','」。').replace('？」','」？').replace('?」','」?').strip()
    
    # 「。」と「？」で分割し、区切り文字もキャプチャ
    parts = re.split(r'([。?？])', text)
    
    split_segments = []
    start_time = segment['start']
    end_time = segment['end']
    total_duration = (end_time - start_time).total_seconds()
    total_chars = len(text)

    current_start = start_time

    # 区切り文字がなくても最後の部分を含める
    for i in range(0, len(parts), 2):
        part = parts[i]
        if i + 1 < len(parts):
            # 区切り文字（「。」や「？」）を取得
            delimiter = parts[i + 1]
            part += delimiter  # 区切り文字を追加
        else:
            delimiter = ''  # 最後の区切り文字がない場合

        if part:
            part_duration = total_duration * len(part) / total_chars
            part_end = current_start + timedelta(seconds=part_duration)
            # 新しいIDを割り当てる
            split_segments.append({
                'id': current_id,
                'start': current_start,
                'end': part_end,
                'text': part
            })
            current_start = part_end
            current_id += 1  # IDをインクリメント

    return split_segments, current_id

def parse_vtt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines=t7.webvtt_rm(lines)
    print("py")
    captions = []
    caption = {'id': None, 'start': None, 'end': None, 'text': ''}
    for line in lines:
        line = line.strip()
        if not line:
            if caption['start'] is not None:
                captions.append(caption)
                caption = {'id': None, 'start': None, 'end': None, 'text': ''}
        elif re.match(r"\d+", line) and caption['id'] is None:
            caption['id'] = line
        elif '-->' in line:
            times = line.split(' --> ')
            caption['start'] = parse_vtt_time(times[0])
            caption['end'] = parse_vtt_time(times[1])
        else:
            caption['text'] += (line + ' ')

    if caption['start'] is not None:
        captions.append(caption)

    return captions

def save_vtt_file(captions, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('WEBVTT\n\n')
        for caption in captions:
            f.write(f"{caption['id']}\n")
            f.write(f"{format_vtt_time(caption['start'])} --> {format_vtt_time(caption['end'])}\n")
            f.write(f"{caption['text'].replace('」。','。」').replace('」？','？」').replace('」?','?」').strip()}\n\n")

def split_vtt_captions(captions):
    split_captions = []
    current_id = 1  # IDの開始値
    for caption in captions:
        # 分割したキャプションと次のIDを取得
        split_segments, current_id = split_vtt_segment(caption, current_id)
        split_captions.extend(split_segments)
    return split_captions

def split_srt_segment(segment, current_index):
    
    text = segment.content.replace('。」','」。').replace('？」','」？').replace('?」','」?').strip()
    
    # 正規表現で「。」と「？」を区切り文字として分割し、区切り文字もキャプチャ
    
    parts = re.split(r'([。？?])', text)
    
    split_segments = []
    start_time = segment.start
    end_time = segment.end
    total_duration = (end_time - start_time).total_seconds()
    total_chars = len(text)

    current_start = start_time

    # 2ステップずつ進んで、パートと区切り文字を処理
    for i in range(0, len(parts), 2):
        part = parts[i]
        if i + 1 < len(parts):
            # 区切り文字（「。」や「？」）を取得
            delimiter = parts[i + 1]
            part += delimiter  # パートに区切り文字を追加
        else:
            delimiter = ''  # 区切り文字がない場合

        if part:
            part_duration = total_duration * len(part) / total_chars
            part_end = current_start + timedelta(seconds=part_duration)
            # 新しいユニークなID（current_index）を割り当てる
            split_segments.append(srt.Subtitle(
                index=current_index,  # 新しいIDを設定
                start=current_start,
                end=part_end,
                content=part.replace('」。','。」').replace('」？','？」').replace('」?','?」')
            ))
            current_start = part_end
            current_index += 1  # IDをインクリメントしてユニークにする

    return split_segments, current_index

def split_srt_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f.read()))

    new_subtitles = []
    current_index = 1  # 新しいIDの開始値
    for subtitle in subtitles:
        # 分割後のセグメントと次のインデックスを取得
        split_segments, current_index = split_srt_segment(subtitle, current_index)
        new_subtitles.extend(split_segments)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(srt.compose(new_subtitles))
def process_files(files):
    results = []

    timestamp_patch = datetime.now().strftime("%Y%m%d%H%M%S")
    temp_dir = os.path.join(tempfile.gettempdir(), f"tempdir_{timestamp_patch}")
    os.makedirs(temp_dir, exist_ok=True)

    for input_file in files:
        base_name = os.path.splitext(input_file)[0]
        base_name = base_name.replace("_ja","")
        print(input_file)
        if input_file.endswith('.vtt'):
            captions = parse_vtt_file(input_file)
            split_captions = split_vtt_captions(captions)
            output_file = f'{base_name}_sp_ja.vtt'
            output_file = os.path.join(temp_dir,output_file)
            save_vtt_file(split_captions, output_file)
            results.append(output_file)

        elif input_file.endswith('.srt'):
            output_file = f'{base_name}_sp_ja.srt'
            output_file= os.path.join(temp_dir,output_file)
            split_srt_file(input_file, output_file)
            results.append(output_file)
        else:
            print(f"Unsupported file format: {input_file}")

    # If there are multiple output files, zip them
    if len(results) > 1:
        zip_file = 'splitted_files.zip'
        zip_file = os.path.join(temp_dir,zip_file)
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            for file in results:
                zipf.write(file, os.path.basename(file))
        results.append(zip_file)

    return results

def clear_files():
    return None, None

def process_and_display(files):
    if not files:
        return None
    results = process_files(files)
    return results


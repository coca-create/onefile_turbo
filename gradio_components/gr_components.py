import gradio as gr
from tab1 import tab1_func as t1
from tab2 import tab2_func as t2

import pandas as pd



def gr_components():

    with gr.Blocks() as UI:
        gr.Markdown(
            """
            <h1 style="color:'darkblue'; font-family :'Arial', sans-serif;font-size:36px;"> PeriOz web　- Transcribe - </h1>
            <p style="color:gray; letter-spacing:0.05em;">OpenAIのfaster-whisperを使っています。字幕の区切りを必ずピリオドにできるのがこのアプリの特徴です。それにより翻訳精度が保たれます。</p>
            """)
        
        ### Gradio-Tab1 ###
                   
        with gr.Tab("文字起こし",elem_classes="Tab1"):
            gr.Markdown("> 字幕ファイル（srtファイル）、テキストファイル2種、Google翻訳用ワード、エクセルファイルが表示されます。Google翻訳用のファイルが必要な場合はアコーディオンを開いてね。") 
            with gr.Row():
                with gr.Column():
                    param1 = gr.File(label="ファイルをアップロードしてね",type="filepath")#,file_types=['mp3','mp4','webm','mkv']
                    with gr.Row():
                        exec_btn = gr.Button("データファイルの作成", variant="primary")
                        t1_clear_Button=gr.Button(value='クリア')
                    param2 = gr.Dropdown(["medium", "large-v1", "large-v2", "large-v3-turbo", "large-v3"], value="large-v2", label="モデルを選ぶ")
                    param3 = gr.Radio(["int8", "float16", "float32","int8_float16","int8_float32"], value="float32", label="演算方法を選ぶ")
                    with gr.Row():
                        param4 = gr.Radio(["英語"], value="英語", label="言語を選ぶ")
                        param0 = gr.Radio(["cuda","cpu"],value="cuda",label="デバイスを選択してください。")

                    param5 = gr.Slider(label="ビームサイズ", value=5, minimum=1, maximum=10, step=1)
                    param6 = gr.Checkbox(label="Vad-Filterの使用", value=True)
                    with gr.Accordion(label="Google翻訳用docs",open=False):
                        doc_download_path=gr.File(label="Wordファイルのダウンロード",file_count="multiple")
                    
                    '''with gr.Accordion(label="Json（ワードスタンプ）", open=False):
                        result_json_file = gr.File(label="JSONファイルをダウンロード")
                        result_json_content = gr.TextArea(label="JSONファイルの内容を表示", autoscroll=False, show_copy_button=True)
                    '''
                with gr.Column():
                    main_files_path = gr.File(label="SRT,TXT(NR,R)ファイルのダウンロード", file_count="multiple")
                    

                    with gr.Column():
                        result_srt_content = gr.TextArea(label="SRTファイルの内容を表示", autoscroll=False, show_copy_button=True)
                        result_txt_nr_content = gr.TextArea(label='「改行なし（NR）」のTXTファイルの内容を表示', autoscroll=False, show_copy_button=True)
                        result_txt_r_content = gr.TextArea(label='ピリオド区切りで「改行した（R）」のTXTファイルの内容を表示', autoscroll=False, show_copy_button=True)


        ### Gradio-Tab2 ###
        
        with gr.Tab("翻訳お手伝い①"):
            gr.Markdown("> 文字起こし直後に行う翻訳ファイル作成を手伝います。翻訳文を貼り付けて日本語ファイルを作ります。SRTファイルを作成するときは、同時に日英字幕が記載されたExcelファイルも作成されます。")
            with gr.Column():
                filename_output=gr.Textbox(label="ファイルのベースネーム",placeholder="ファイル名が空、または間違っているときは記入してね。")
                extension_choices = gr.CheckboxGroup(["srt", "txt(nr)", "txt(r)"],value=["srt"], label="ファイルの種類を選んで下さい。")
            with gr.Column():
                gr.Markdown("ここに翻訳コピーを貼り付けます。")
                with gr.Row():
                    translate_srt = gr.TextArea(label="Translate Content for _ja.srt", visible=True)
                    translate_nr_txt = gr.TextArea(label="Translate Content for _NR_ja.txt", visible=False)
                    translate_r_txt = gr.TextArea(label="Translate Content for _R_ja.txt", visible=False)
                    dummy=gr.Textbox(label="後で非表示" ,visible=False)

                with gr.Row():
                    html_srt = gr.HTML(visible=True)
                    html_nr_txt = gr.HTML(visible=False)
                    html_r_txt= gr.HTML(visible=False)
                with gr.Accordion(label="英語dataframe",open=False):
                    gr_components_df=gr.HTML()

            with gr.Row():
                generate_files_button = gr.Button("日本語ファイルの生成",variant="primary")
                t2_clear_button=gr.Button("クリア")
            download_translated_files = gr.File()
            button2_df=gr.HTML()

        def update_visibility(extensions):
            return {
            translate_srt: gr.update(visible="srt" in extensions),
            translate_nr_txt: gr.update(visible="txt(nr)" in extensions),
            translate_r_txt: gr.update(visible="txt(r)" in extensions),
            html_srt:gr.update(visible="srt" in extensions),
            html_nr_txt:gr.update(visible="txt(nr)" in extensions),
            html_r_txt:gr.update(visible="txt(r)" in extensions)
            }




 


        ##クリアボタン追加分をまとめる。
        def t1_clear():
            empty_html_table = pd.DataFrame({'1': [''], '2': [''], '3': ['']}).to_html(index=False)
            return None,"","","",[],[],"","","","","",empty_html_table
        def t2_clear():
            return "","","",[],pd.DataFrame({'1': [''], '2': [''],'3': ['']})
        def t4_clear():
            empty_html_table = pd.DataFrame({'1': [''], '2': [''], '3': ['']}).to_html(index=False)
            return None,"","",[],[],empty_html_table

        '''def t6_clear():
            return None,None,[]'''
        
        def t7_clear():
            empty_html_table = pd.DataFrame({'1': [''], '2': [''], '3': ['']}).to_html(index=False)
            return None,None,None,None,None,None,empty_html_table

        def param1_change_clear():
            return None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None

        ### Tab1 イベントリスナー ###
        param1.change(fn=param1_change_clear,
                      inputs=[],
                      outputs=[result_srt_content,result_txt_nr_content,result_txt_r_content
                               ,main_files_path,doc_download_path,html_srt,html_nr_txt,html_r_txt,filename_output,dummy,gr_components_df,
                               translate_srt,translate_nr_txt,translate_r_txt,download_translated_files,button2_df])
        exec_btn.click(
            fn=t1.transcribe,
            inputs=[param1, param2, param3, param4, param5, param6,param0],
            outputs=[result_srt_content,result_txt_nr_content, result_txt_r_content, main_files_path,doc_download_path,html_srt,html_nr_txt,html_r_txt,filename_output,dummy,gr_components_df])
        
        t1_clear_Button.click(
            fn=t1_clear,inputs=[],outputs=[param1,result_srt_content,result_txt_nr_content,result_txt_r_content,main_files_path,doc_download_path,html_srt,html_nr_txt,html_r_txt,filename_output,dummy,gr_components_df]
        )
        ### Tab2 イベントリスナー ###
        
        extension_choices.change(fn=update_visibility, 
                                inputs=extension_choices,
                                outputs=[translate_srt, translate_nr_txt, translate_r_txt,html_srt,html_nr_txt,html_r_txt])
        
        generate_files_button.click(
            fn=t2.create_translate_files,
            inputs=[filename_output, 
                    translate_srt,
                    translate_nr_txt,
                    translate_r_txt, 
                    extension_choices,
                    dummy],
            outputs=[download_translated_files,button2_df]) 
        t2_clear_button.click(fn=t2_clear,inputs=[],outputs=[translate_srt,translate_nr_txt,translate_r_txt,download_translated_files,button2_df])

        return UI

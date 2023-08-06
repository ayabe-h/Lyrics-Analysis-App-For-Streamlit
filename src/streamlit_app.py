import io
import MeCab
import pandas as pd
import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import re
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# ページのレイアウトを設定
st.set_page_config(
    page_title="テキスト可視化",
    layout="wide", # wideにすると横長なレイアウトに
    initial_sidebar_state="expanded"
)

# タイトルの設定
st.title("テキスト可視化")

# サイドバーにアップロードファイルのウィジェットを表示
st.sidebar.markdown("# ファイルアップロード")
uploaded_file = st.sidebar.file_uploader(
    "テキストファイルをアップロードしてください", type="txt"
)

# ワードクラウド、出現頻度表の各処理をサイドバーに表示
st.sidebar.markdown("# 可視化のオプション")
if uploaded_file is not None:
    # 処理の選択
    option = st.sidebar.selectbox(
        "処理の種類を選択してください", ["ワードクラウド", "出現頻度表", "極性表"]
    )
    # ワードクラウドの表示
    if option == "ワードクラウド":
        pos_options = ["名詞", "形容詞", "動詞", "副詞", "助詞", "助動詞", "接続詞", "感動詞", "連体詞", "記号", "未知語"]
        # マルチセレクトボックス
        selected_pos = st.sidebar.multiselect("品詞選択", pos_options, default=["名詞"])
        if st.sidebar.button("生成"):
            st.markdown("## ワードクラウド")
            with st.spinner("Generating..."):
                io_string = io.StringIO(uploaded_file.getvalue().decode("UTF-8"))
                text = io_string.read()
                tagger = MeCab.Tagger()
                node = tagger.parseToNode(text)
                words = []
                while node:
                    if node.surface.strip() != "":
                        word_type = node.feature.split(",")[0]
                        if word_type in selected_pos: # 対象外の品詞はスキップ
                            words.append(node.surface)
                    node = node.next
                word_count = Counter(words)
                wc = WordCloud(
                    width=800,
                    height=800,
                    background_color="white",
                    font_path="meiryob.ttc", # Fontを指定
                )
                # ワードクラウドを作成
                wc.generate_from_frequencies(word_count)
                # ワードクラウドを表示
                st.image(wc.to_array())
    
    # 出現頻度表の表示
    elif option == "出現頻度表":
        pos_options = ["名詞", "形容詞", "動詞", "副詞", "助詞", "助動詞", "接続詞", "感動詞", "連体詞", "記号", "未知語"]
        # マルチセレクトボックス
        selected_pos = st.sidebar.multiselect("品詞選択", pos_options, default=pos_options)
        if st.sidebar.button("生成"):
            st.markdown("## 出現頻度表")
            with st.spinner("Generating..."):
                io_string = io.StringIO(uploaded_file.getvalue().decode("UTF-8"))
                text = io_string.read()
                tagger = MeCab.Tagger()
                node = tagger.parseToNode(text)

                # 品詞ごとに出現単語と出現回数をカウント
                pos_word_count_dict = {}
                while node:
                    pos = node.feature.split(",")[0]
                    if pos in selected_pos:
                        if pos not in pos_word_count_dict:
                            pos_word_count_dict[pos] = {}
                        if node.surface.strip() != "":
                            word = node.surface
                            if word not in pos_word_count_dict[pos]:
                                pos_word_count_dict[pos][word] = 1
                            else:
                                pos_word_count_dict[pos][word] += 1
                    node = node.next

                # カウント結果を表にまとめる
                pos_dfs = []
                for pos in selected_pos:
                    if pos in pos_word_count_dict:
                        df = pd.DataFrame.from_dict(pos_word_count_dict[pos], orient="index", columns=["出現回数"])
                        df.index.name = "出現単語"
                        df = df.sort_values("出現回数", ascending=False)
                        pos_dfs.append((pos, df))

                # 表を表示
                for pos, df in pos_dfs:
                    st.write(f"【{pos}】")
                    st.dataframe(df, 400, 400)      
    elif option == '極性表':
        if st.sidebar.button("生成"):
            st.markdown("## 極性表")
            with st.spinner("Generating..."):
                # 感情値辞書の読み込み
                try:
                    pndic = pd.read_csv("pn_ja.csv", encoding="shift-jis", names=["word_type_score"])
                except FileNotFoundError:
                    st.error("感情値辞書ファイル (pn_ja.csv) が見つかりません。")

                # 語と感情値を抽出
                pndic["split"] = pndic["word_type_score"].str.split(":")
                pndic["word"] = pndic["split"].str.get(0)
                pndic["score"] = pndic["split"].str.get(-1)  # -1を指定して最後の要素を取得

                # score列の値を数値に変換
                pndic["score"] = pd.to_numeric(pndic["score"], errors="coerce")  # エラーが発生した場合はNaNに設定

                # dict型に変換
                pn_dict = pndic.set_index("word")["score"].to_dict()
                # テキストファイルを読み込む
                io_string = io.StringIO(uploaded_file.getvalue().decode("UTF-8"))
                text = io_string.read()

                def mecab_parse(text):
                    tagger = MeCab.Tagger()
                    node = tagger.parse(text)
                    lines = node.split("\n")
                    dilist = []
                    for word in lines:
                        l = re.split('\t|,', word)
                        if len(l) > 8:
                            d = {'Surface': l[0], 'POS': l[1], 'BaseForm': l[8]}
                            dilist.append(d)
                    return dilist

                dilist = mecab_parse(text)

                def add_pnvalue(dilist_old, pn_dict):
                    dilist_new = []
                    for word in dilist_old:
                        base = word['BaseForm']
                        if base in pn_dict:
                            pn = float(pn_dict[base])
                        else:
                            pn = 'notfound'
                        word['PN'] = pn
                        dilist_new.append(word)
                    return dilist_new

                def get_mean(diclist):
                    pn_list = []
                    for word in diclist:
                        pn = word['PN']
                        if pn!= 'notfound':
                            pn_list.append(pn)
                            if len(pn_list)>0:
                                pnmean = np.mean(pn_list)
                            else:
                                pnmean = 0
                                
                            return pnmean
                
                dilist_with_pn = add_pnvalue(dilist, pn_dict)
                

                # dilist_with_pnに感情値が存在する場合のみ、数値のリストを取得
                pn_values = [word['PN'] for word in dilist_with_pn if 'PN' in word and isinstance(word['PN'], (int, float))]

                
                sns.histplot(pd.Series(pn_values), bins=20, kde=False, color='blue', edgecolor='black')
                plt.xlabel('score')
                plt.ylabel('frequency')
                plt.grid(True)
                
                # ヒストグラムのfigureオブジェクトを取得
                fig = plt.gcf()

                # ヒストグラムをstreamlit上に表示
                st.pyplot(fig)

                # 平均感情値を計算
                pn_mean = get_mean(dilist_with_pn)


                # 平均感情値を表示
                st.write("平均感情値:", pn_mean)

else:
    # テキスト未アップロード時の処理
    st.write("テキストファイルをアップロードしてください。")

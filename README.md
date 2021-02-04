## 【学生限定】SIGNATE 22卒インターン選考コンペティション
- 結果: public 1st -> private 6th
- コンペURL: https://signate.jp/competitions/402

## public 1st, private 6th place solution

### 1. 特徴量
  - html content
    - 前処理なし,html tag のみ, text のみなどに処理して以下の特徴量を作成
    - doc2vec
    - tfidf -> svd
    - count -> svd
    - html tag (種類)数, 文字数などbasicな特徴量
    
  - html content 以外
    - raw: duration, goal
    - binning: duration & goal
    - 組み合わせ特徴量: duration, goal, category, country, binning など
    - ranking特徴量:  duration, goal, category, country, binning, 組み合わせ特徴量など
    - 集約特徴量: 組み合わせ特徴量, basicなtext特徴量, カテゴリ変数などに対し,「max, min, mean, std, count, max-min, z-socre」で集約
    
  - 最終的に3500程度の特徴量を作ってそこからを600個を選択

### 2. model
  - lightgbm (10 folds)
  - 7 seed average
  
### 3. 効いた取り組み, 特徴量
  - html content 情報を用いた aggregation が効いた. 集約方法としては、std と z-scoreの重要度が高くこれも良く効いていた.
  - tag情報のみから抽出した特徴量はかなり精度向上に貢献した. カウントベースのものを次元削減するだけでも効果があった.
  - textのみからの特徴量も精度向上に貢献していたので、bert系で特徴量を作ったりしてもよかったと思う.
  
### 4. 効かなかった取り組み, 特徴量
  - text vector からクラスタリングしたものの集約特徴量など入れてみたが, 重要度はかなり高かったが精度向上には貢献しなかった.
  - 自分の特徴量・モデルではあまりスタッキングなどアンサンブルの効果がなかった. privateだとあったのかもしれない. 
  - target encoding

### 5. 実験管理
  - 試験的に、1実験を1スクリプト内で全て納めて、ログや特徴量重要度を実験ごとにフォルダごと動的に作成してみた. けっこうよかった.
  
  ![キャプチャ](https://user-images.githubusercontent.com/64417843/106902581-78f6a880-66a5-11eb-9d63-a8b3915e8adb.PNG)
  
### 6. 感想
  難易度てきにも難し過ぎないコンペで楽しむことができました!     
  特徴量がほぼ同じサブでもprivateでは結構ブレがあったので、サブ最終選択が難しかったと感じました.      
  参加者の皆さんお疲れ様でした！運営さんありがとうございました1
  

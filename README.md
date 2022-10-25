# posture_detection_demo

## 概要
複数の動的姿勢検出ツールの精度を視覚的に表示するツール．また，背景差分やRealSenseの深度計測を使った背景へのマスク処理を行うことで，姿勢検出の精度向上を検証できる．リアルタイムでの描画の他に録画機能も実装しており，録画後も深度の基準値を変化させて描画できる．

 ![screenshot](https://raw.githubusercontent.com/hakoshi-normal/md_images/main/realsenseGUI_view.png "画面")


## 要件
このコードは，Windows10でテストしている．
その他テスト環境での依存ツールは以下．
* Google Chrome (v106)
* Python3.9（miniconda）
* Pythonライブラリ各種（requirements.txtを参照）

## インストール
ブラウザ，Pythonをインストールしておく．  
このリポジトリのクローンを作成する．例：
```shell
git clone https://github.com/hakoshi-normal/posture_detection_demo.git
```

任意の環境下で以下のコマンドを実行して依存ライブラリをインストールする．
```shell
pip install -r requirements.txt
```

## 実行
予めカメラデバイスを接続しておき，以下のコマンドを実行する．
```shell
python run.py
```
RealSenseを優先的に利用する．RealSenseが検出されない場合，その他のカメラデバイスを利用する．

## その他
* 録画データは未圧縮状態で保存されるため長時間の録画は非推奨．
* 録画データは生成されるsaveディレクトリ内に保存される．
* FPS値は姿勢検出ツール及び背景マスク手法とその設定値を変更するとその都度リセットされる．  
* スパゲッティコードです．

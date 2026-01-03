import numpy as np

# .npyファイルのヘッダー情報を詳しく確認
file_path = 'models/tts/style_vectors.npy'

# ファイルをバイナリモードで開いてヘッダーを直接読む
with open(file_path, 'rb') as f:
    # NPYファイルの最初の6バイトはマジックナンバー
    magic = f.read(6)
    print(f"Magic: {magic}")

    # バージョン情報（メジャー、マイナー）
    major_version = np.frombuffer(f.read(1), dtype=np.uint8)[0]
    minor_version = np.frombuffer(f.read(1), dtype=np.uint8)[0]
    print(f"NPY Format Version: {major_version}.{minor_version}")

    # ヘッダー長
    header_len = np.frombuffer(f.read(2), dtype=np.uint16)[0]
    print(f"Header Length: {header_len}")

    # ヘッダー情報（辞書形式）
    header = f.read(header_len)
    print(f"Header: {header.decode('latin1')}")

# 簡単な方法：ファイルをロードして基本情報を確認
try:
    data = np.load(file_path)
    print(f"\nデータ型: {data.dtype}")
    print(f"形状: {data.shape}")
    print(f"使用中のnumpyバージョン: {np.__version__}")
except Exception as e:
    print(f"エラー: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォントのサポート
from datetime import datetime
import os

# ページの設定
st.set_page_config(
    page_title="SageMaker Canvas時系列モデル分析ダッシュボード",
    page_icon="📊",
    layout="wide"
)

# タイトルと説明
st.title("SageMaker Canvas時系列モデル分析ダッシュボード")
st.markdown("Amazon SageMaker Canvasで作成した時系列モデルの分析結果を表示します。")

# データの読み込み関数
@st.cache_data
def load_data():
    # データパスを指定（Streamlit Cloudでも動作するように絶対パスを使用）
    import os
    
    # アプリのルートディレクトリを取得
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # データパスを絶対パスで指定
    train_path = os.path.join(root_dir, "data", "train", "SKU_rev_train.csv")
    test_path = os.path.join(root_dir, "data", "test", "SKU需要予測_test.csv")
    result_path = os.path.join(root_dir, "data", "result", "result_summary.csv")
    
    # データの読み込み
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    result_df = pd.read_csv(result_path)
    
    # 日付列を日付型に変換
    train_df['Date'] = pd.to_datetime(train_df['Date'], format='%Y/%m/%d')
    test_df['Date'] = pd.to_datetime(test_df['Date'], format='%Y/%m/%d')
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    
    # 学習データとテストデータを結合
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 予測結果を結合
    # result_dfには予測値（p10, p50, p90, mean）が含まれている
    # p50（中央値）を予測値として使用
    df_with_prediction = df.copy()
    
    # 予測結果をマージ
    prediction_data = result_df[['Item_name', 'Date', 'p50']].rename(columns={'p50': 'Prediction'})
    df_with_prediction = pd.merge(
        df_with_prediction, 
        prediction_data, 
        on=['Item_name', 'Date'], 
        how='left'
    )
    
    return df_with_prediction

# 指標計算関数
def calculate_metrics(actual, predicted):
    # NaNを除外
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    # MAPEの計算（ゼロ除算を回避）
    mask = actual != 0  # ゼロでない値のマスク
    if np.any(mask):
        # ゼロでない値のみでMAPEを計算
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan  # すべてゼロの場合はNaNを返す
    
    # 過剰在庫率（予測>実績の割合）
    overstock_rate = np.mean(predicted > actual) * 100
    
    # 欠品率（予測<実績の割合）
    stockout_rate = np.mean(predicted < actual) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Overstock Rate': overstock_rate,
        'Stockout Rate': stockout_rate
    }

# 特徴量重要度の計算
def get_feature_importance(df, sku_id):
    # 選択した商品のデータをフィルタリング
    sku_data = df[df['Item_name'] == sku_id].copy()
    
    # 特徴量の列名を取得
    feature_cols = [
        'Price', 'coupon_rate', 'holiday', 'tempreture',
        'precipitation', 'Sales_lag_1', 'Sales_lag_7', 'CPI',
        'Closed_flag', 'market_flag'
    ]
    
    # 特徴量の重要度を計算（相関係数の絶対値を使用）
    importances = []
    for col in feature_cols:
        if col in sku_data.columns:
            # NaNを除外して相関係数を計算
            corr = sku_data[['Sales', col]].dropna().corr().iloc[0, 1]
            importances.append(abs(corr))
        else:
            importances.append(0)
    
    # 特徴量重要度のデータフレームを作成
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    
    # 重要度でソート
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # 重要度を0-1の範囲に正規化
    if importance_df['Importance'].max() > 0:
        importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].max()
    
    return importance_df

# 異常検出関数
def detect_anomalies(actual, predicted, threshold=2):
    # NaNを除外
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    residuals = actual_clean - predicted_clean
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # 標準偏差の閾値を超える誤差を持つポイントを特定
    anomalies = np.abs(residuals - mean_residual) > threshold * std_residual
    
    # 元の配列と同じサイズの結果配列を作成（デフォルトはFalse）
    full_anomalies = np.zeros(len(actual), dtype=bool)
    
    # マスクされたインデックスに対応する位置に異常値フラグを設定
    full_anomalies[mask] = anomalies
    
    return full_anomalies

# 売上予測と実績のプロット関数
def plot_sales_prediction(df, sku_id, days=30):
    # 選択した商品のデータをフィルタリング
    sku_data = df[df['Item_name'] == sku_id].copy()
    
    # 予測値がある行のみを抽出
    sku_data = sku_data.dropna(subset=['Prediction'])
    
    # 最新の日付から指定日数分のデータを抽出
    latest_data = sku_data.sort_values('Date', ascending=False).head(days)
    plot_data = latest_data.sort_values('Date')
    
    # プロット
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data['Date'], 
        y=plot_data['Sales'],
        mode='lines+markers',
        name='実績',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=plot_data['Date'], 
        y=plot_data['Prediction'],
        mode='lines+markers',
        name='予測',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='日次売上高: 実績 vs 予測（最新{}日分）'.format(days),
        xaxis_title='日付',
        yaxis_title='売上高',
        legend_title='データ種別',
        height=500
    )
    
    return fig

# 特徴量の部分依存プロット
def plot_partial_dependence(df, sku_id, feature):
    # 選択した商品のデータをフィルタリング
    sku_data = df[df['Item_name'] == sku_id].copy()
    
    # 特徴量の値でグループ化して平均売上を計算
    if feature in sku_data.columns:
        # 数値型の場合はビン分割
        if np.issubdtype(sku_data[feature].dtype, np.number):
            # ビン分割（10分位）
            sku_data['bin'] = pd.qcut(sku_data[feature], 10, duplicates='drop')
            grouped = sku_data.groupby('bin')['Sales'].mean().reset_index()
            # ビンの中央値を取得
            grouped['bin_mid'] = grouped['bin'].apply(lambda x: x.mid)
            
            # プロット
            fig = px.line(
                grouped, 
                x='bin_mid', 
                y='Sales',
                markers=True,
                title=f'特徴量の部分依存: {feature}',
                labels={'bin_mid': feature, 'Sales': '平均売上'}
            )
        else:
            # カテゴリ型の場合はそのままグループ化
            grouped = sku_data.groupby(feature)['Sales'].mean().reset_index()
            
            # プロット
            fig = px.bar(
                grouped, 
                x=feature, 
                y='Sales',
                title=f'特徴量の部分依存: {feature}',
                labels={feature: feature, 'Sales': '平均売上'}
            )
    else:
        # 特徴量がない場合はダミーのグラフを返す
        fig = go.Figure()
        fig.update_layout(
            title=f'特徴量の部分依存: {feature} (データなし)',
            xaxis_title=feature,
            yaxis_title='平均売上'
        )
    
    return fig

# 需要変動要因の円グラフ関数
def plot_demand_factors(df, sku_id):
    # 選択した商品のデータをフィルタリング
    sku_data = df[df['Item_name'] == sku_id].copy()
    
    # 特徴量の列名を取得
    feature_cols = [
        'Price', 'coupon_rate', 'holiday', 'tempreture',
        'precipitation', 'Sales_lag_1', 'Sales_lag_7', 'CPI',
        'Closed_flag', 'market_flag'
    ]
    
    # 特徴量の重要度を計算（相関係数の絶対値を使用）
    factors = {}
    for col in feature_cols:
        if col in sku_data.columns:
            # NaNを除外して相関係数を計算
            corr = sku_data[['Sales', col]].dropna().corr().iloc[0, 1]
            factors[col] = abs(corr)
        else:
            factors[col] = 0
    
    # 特徴量名を日本語に変換
    feature_names_ja = {
        'Price': '価格',
        'coupon_rate': 'クーポン率',
        'holiday': '休日',
        'tempreture': '気温',
        'precipitation': '降水量',
        'Sales_lag_1': '前日売上',
        'Sales_lag_7': '前週売上',
        'CPI': '消費者物価指数',
        'Closed_flag': '休業フラグ',
        'market_flag': '市場フラグ'
    }
    
    # 日本語名に変換
    factors_ja = {feature_names_ja.get(k, k): v for k, v in factors.items()}
    
    # 合計が100%になるように正規化
    total = sum(factors_ja.values())
    if total > 0:
        factors_ja = {k: v/total*100 for k, v in factors_ja.items()}
    
    fig = px.pie(
        values=list(factors_ja.values()),
        names=list(factors_ja.keys()),
        title='需要変動要因の内訳',
        hole=0.3
    )
    
    return fig

# 交互作用効果の計算と可視化
def calculate_interactions(df, sku_id):
    # 選択した商品のデータをフィルタリング
    sku_data = df[df['Item_name'] == sku_id].copy()
    
    # 特徴量の列名を取得
    feature_cols = [
        'Price', 'coupon_rate', 'holiday', 'tempreture',
        'precipitation', 'Sales_lag_1', 'Sales_lag_7', 'CPI',
        'Closed_flag', 'market_flag'
    ]
    
    # 存在する特徴量のみを使用
    available_features = [col for col in feature_cols if col in sku_data.columns]
    
    # 交互作用の計算
    interactions = []
    
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            feat1 = available_features[i]
            feat2 = available_features[j]
            
            # 両方の特徴量でNaNがない行のみを使用
            valid_data = sku_data.dropna(subset=[feat1, feat2, 'Sales'])
            
            if len(valid_data) > 10:  # 十分なデータがある場合のみ
                # 特徴量を2値化（中央値で分割）
                valid_data[f'{feat1}_high'] = valid_data[feat1] > valid_data[feat1].median()
                valid_data[f'{feat2}_high'] = valid_data[feat2] > valid_data[feat2].median()
                
                # 4つのグループの平均売上を計算
                g1 = valid_data[(~valid_data[f'{feat1}_high']) & (~valid_data[f'{feat2}_high'])]['Sales'].mean()
                g2 = valid_data[(valid_data[f'{feat1}_high']) & (~valid_data[f'{feat2}_high'])]['Sales'].mean()
                g3 = valid_data[(~valid_data[f'{feat1}_high']) & (valid_data[f'{feat2}_high'])]['Sales'].mean()
                g4 = valid_data[(valid_data[f'{feat1}_high']) & (valid_data[f'{feat2}_high'])]['Sales'].mean()
                
                # 交互作用効果の計算
                # (g4 - g3) - (g2 - g1) = g4 - g3 - g2 + g1
                interaction_effect = g4 - g3 - g2 + g1
                
                # 交互作用効果の絶対値を正規化
                interactions.append({
                    'Factor1': feat1,
                    'Factor2': feat2,
                    'Effect': abs(interaction_effect)
                })
    
    # 交互作用効果でソート
    interactions_df = pd.DataFrame(interactions).sort_values('Effect', ascending=False)
    
    # 上位5つの交互作用を取得
    top_interactions = interactions_df.head(5)
    
    # 特徴量名を日本語に変換
    feature_names_ja = {
        'Price': '価格',
        'coupon_rate': 'クーポン率',
        'holiday': '休日',
        'tempreture': '気温',
        'precipitation': '降水量',
        'Sales_lag_1': '前日売上',
        'Sales_lag_7': '前週売上',
        'CPI': '消費者物価指数',
        'Closed_flag': '休業フラグ',
        'market_flag': '市場フラグ'
    }
    
    # 日本語名に変換
    top_interactions['Factor1_ja'] = top_interactions['Factor1'].map(lambda x: feature_names_ja.get(x, x))
    top_interactions['Factor2_ja'] = top_interactions['Factor2'].map(lambda x: feature_names_ja.get(x, x))
    
    # 要因ペアのラベルを作成
    top_interactions['Factor1 + Factor2'] = top_interactions['Factor1_ja'] + ' × ' + top_interactions['Factor2_ja']
    
    # 効果の最大値で正規化
    if len(top_interactions) > 0 and top_interactions['Effect'].max() > 0:
        top_interactions['Effect'] = top_interactions['Effect'] / top_interactions['Effect'].max()
    
    return top_interactions

# 主要な交互作用効果の可視化
def plot_interactions(interactions_df):
    if len(interactions_df) == 0:
        # データがない場合はダミーのグラフを返す
        fig = go.Figure()
        fig.update_layout(
            title='主要な交互作用効果 (データなし)',
            xaxis_title='交互作用の強さ',
            yaxis_title='要因ペア'
        )
        return fig
    
    fig = px.bar(
        interactions_df,
        x='Effect',
        y='Factor1 + Factor2',
        orientation='h',
        text='Effect',
        title='主要な交互作用効果',
        labels={'Effect': '交互作用の強さ', 'Factor1 + Factor2': '要因ペア'}
    )
    
    # グラフの更新
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(height=400)
    
    return fig

# 価格とマーケティングの交互作用グラフ
def plot_price_marketing_interaction(df, sku_id):
    # 選択した商品のデータをフィルタリング
    sku_data = df[df['Item_name'] == sku_id].copy()
    
    # 価格とクーポン率の両方がある場合
    if 'Price' in sku_data.columns and 'coupon_rate' in sku_data.columns:
        # 価格を分類（重複値や少ないユニーク値に対応）
        try:
            # まずqcutを試す（等頻度ビン）
            sku_data['price_level'] = pd.qcut(sku_data['Price'], 3, labels=['低', '中', '高'], duplicates='drop')
        except ValueError:
            # qcutが失敗した場合はcutを使用（等間隔ビン）
            try:
                sku_data['price_level'] = pd.cut(sku_data['Price'], 3, labels=['低', '中', '高'])
            except ValueError:
                # それも失敗した場合は、中央値で2分割
                median = sku_data['Price'].median()
                sku_data['price_level'] = pd.cut(
                    sku_data['Price'],
                    bins=[sku_data['Price'].min()-0.1, median, sku_data['Price'].max()+0.1],
                    labels=['低', '高']
                )
        
        # クーポン率を分類
        # クーポン率が0の場合が多いため、特別に処理
        sku_data['coupon_level'] = 'なし'
        non_zero_mask = sku_data['coupon_rate'] > 0
        if non_zero_mask.sum() > 0:
            try:
                # まずqcutを試す（等頻度ビン）
                sku_data.loc[non_zero_mask, 'coupon_level'] = pd.qcut(
                    sku_data.loc[non_zero_mask, 'coupon_rate'],
                    2,
                    labels=['小規模', '大規模'],
                    duplicates='drop'
                )
            except ValueError:
                # qcutが失敗した場合はcutを使用（等間隔ビン）
                try:
                    sku_data.loc[non_zero_mask, 'coupon_level'] = pd.cut(
                        sku_data.loc[non_zero_mask, 'coupon_rate'],
                        2,
                        labels=['小規模', '大規模']
                    )
                except ValueError:
                    # それも失敗した場合は、中央値で2分割
                    median = sku_data.loc[non_zero_mask, 'coupon_rate'].median()
                    sku_data.loc[non_zero_mask, 'coupon_level'] = pd.cut(
                        sku_data.loc[non_zero_mask, 'coupon_rate'],
                        bins=[sku_data.loc[non_zero_mask, 'coupon_rate'].min()-0.001,
                              median,
                              sku_data.loc[non_zero_mask, 'coupon_rate'].max()+0.001],
                        labels=['小規模', '大規模']
                    )
        
        # グループごとの平均売上を計算
        grouped = sku_data.groupby(['price_level', 'coupon_level'])['Sales'].mean().reset_index()
        
        # グラフ作成
        fig = px.line(
            grouped, 
            x='price_level', 
            y='Sales', 
            color='coupon_level',
            markers=True,
            title='価格とクーポン率の交互作用効果',
            labels={'price_level': '価格レベル', 'Sales': '平均売上', 'coupon_level': 'クーポン率'}
        )
    else:
        # データがない場合はダミーのグラフを返す
        fig = go.Figure()
        fig.update_layout(
            title='価格とクーポン率の交互作用効果 (データなし)',
            xaxis_title='価格レベル',
            yaxis_title='平均売上'
        )
    
    return fig

# アプリケーションのメイン関数
def main():
    # データ読み込み
    df = load_data()
    
    # サイドバー
    st.sidebar.header('設定')
    
    # 商品選択
    sku_options = sorted(df['Item_name'].unique())
    selected_sku = st.sidebar.selectbox('商品を選択', sku_options)
    
    # 選択した商品のデータをフィルタリング
    sku_data = df[df['Item_name'] == selected_sku]
    
    # データ期間のスライダー
    date_range = st.sidebar.slider(
        '分析期間',
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date(),
        value=(df['Date'].min().date(), df['Date'].max().date())
    )
    
    # 日付でフィルタリング
    filtered_data = sku_data[
        (sku_data['Date'].dt.date >= date_range[0]) & 
        (sku_data['Date'].dt.date <= date_range[1])
    ]
    
    # 1. データ概要と評価指標セクション
    st.header('1. データ概要と評価指標')
    
    # データ概要表示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 商品IDと商品名のマッピング
        sku_name_mapping = {
            '0': 'カップラーメン しお',
            '1': 'ハイチュウ グレープ',
            '2': 'ポテトチップス コンソメパンチ'
        }
        product_name = sku_name_mapping.get(selected_sku, selected_sku)
        st.metric('商品', f"{product_name} (ID: {selected_sku})")
    
    with col2:
        st.metric('データ期間', f"{filtered_data['Date'].min().date()} 〜 {filtered_data['Date'].max().date()}")
    
    with col3:
        st.metric('レコード数', len(filtered_data))
    
    # 予測値がある行のみを抽出して評価指標を計算
    prediction_data = filtered_data.dropna(subset=['Prediction'])
    
    if len(prediction_data) > 0:
        # モデル評価指標の計算
        metrics = calculate_metrics(prediction_data['Sales'].values, prediction_data['Prediction'].values)
        
        # 評価指標の表示
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric('RMSE', f"{metrics['RMSE']:.2f}")
        
        with col2:
            st.metric('MAE', f"{metrics['MAE']:.2f}")
        
        with col3:
            st.metric('MAPE', f"{metrics['MAPE']:.2f}%")
        
        with col4:
            st.metric('過剰在庫率', f"{metrics['Overstock Rate']:.2f}%")
        
        with col5:
            st.metric('欠品率', f"{metrics['Stockout Rate']:.2f}%")
    else:
        st.warning("選択した期間に予測データがありません。")
    
    # 2. 時系列可視化セクション
    st.header('2. 時系列可視化')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 日次売上高の実績vs予測グラフ
        if len(prediction_data) > 0:
            days_to_show = st.slider('表示日数', min_value=7, max_value=90, value=30)
            sales_fig = plot_sales_prediction(filtered_data, selected_sku, days=days_to_show)
            st.plotly_chart(sales_fig, use_container_width=True)
        else:
            st.info("選択した期間に予測データがありません。")
    
    with col2:
        # 特徴量重要度のグラフ
        feature_importance = get_feature_importance(filtered_data, selected_sku)
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='特徴量重要度',
            labels={'Importance': '重要度', 'Feature': '特徴量'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # 特徴量の部分依存（重要度が高い特徴量3つ）
    st.subheader('特徴量の部分依存（重要度が高い特徴量3つ）')
    
    # 上位3つの特徴量を取得
    top_features = feature_importance.head(3)['Feature'].tolist()
    
    # 3つの列に分けて表示
    cols = st.columns(3)
    
    for i, feature in enumerate(top_features):
        with cols[i]:
            pdp_fig = plot_partial_dependence(filtered_data, selected_sku, feature)
            st.plotly_chart(pdp_fig, use_container_width=True)
    
    # 3. 需要要因分析セクション
    st.header('3. 需要要因分析')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 需要変動要因の円グラフ
        demand_factors_fig = plot_demand_factors(filtered_data, selected_sku)
        st.plotly_chart(demand_factors_fig, use_container_width=True)
    
    with col2:
        # 価格とマーケティングの交互作用ライングラフ
        price_marketing_fig = plot_price_marketing_interaction(filtered_data, selected_sku)
        st.plotly_chart(price_marketing_fig, use_container_width=True)
    
    # 4. 交互作用効果セクション
    st.header('4. 交互作用効果')
    
    # 交互作用効果の計算
    interactions_df = calculate_interactions(filtered_data, selected_sku)
    
    # 主要な交互作用効果の視覚化
    interactions_fig = plot_interactions(interactions_df)
    st.plotly_chart(interactions_fig, use_container_width=True)
    
    # 交互作用の解釈
    if len(interactions_df) > 0:
        st.subheader('交互作用効果の解釈')
        
        # 上位3つの交互作用について解釈を表示
        for i, row in interactions_df.head(3).iterrows():
            st.write(f"- **{row['Factor1 + Factor2']}**: これらの要因が組み合わさると、単独の効果よりも大きな影響が売上に現れます。")
    
    # 5. 異常検出セクション
    st.header('5. 異常検出')
    
    if len(prediction_data) > 0:
        # 異常値の検出
        anomalies = detect_anomalies(prediction_data['Sales'], prediction_data['Prediction'])
        anomaly_data = prediction_data[anomalies].copy()
        
        # 異常値テーブルの表示
        if len(anomaly_data) > 0:
            st.subheader(f'検出された異常値 ({len(anomaly_data)}件)')
            
            # 表示用にデータを整形
            display_data = anomaly_data.copy()
            display_data['Date'] = display_data['Date'].dt.date
            display_data['Error'] = display_data['Sales'] - display_data['Prediction']
            display_data['Error_Pct'] = (display_data['Error'] / display_data['Sales']) * 100
            
            # 表示するカラムを選択
            columns_to_display = ['Date', 'Sales', 'Prediction', 'Error', 'Error_Pct', 'Price', 
                                'holiday', 'tempreture', 'precipitation']
            
            # 存在するカラムのみを表示
            columns_to_display = [col for col in columns_to_display if col in display_data.columns]
            
            st.dataframe(
                display_data[columns_to_display].sort_values('Date', ascending=False),
                hide_index=True,
                column_config={
                    'Date': '日付',
                    'Sales': '実績売上',
                    'Prediction': '予測売上',
                    'Error': '誤差',
                    'Error_Pct': st.column_config.NumberColumn(
                        '誤差率 (%)',
                        format="%.2f%%"
                    ),
                    'Price': '価格',
                    'holiday': '休日',
                    'tempreture': '気温',
                    'precipitation': '降水量'
                }
            )
            
            # 異常値の視覚化
            anomaly_fig = go.Figure()
            
            # すべてのデータポイント
            anomaly_fig.add_trace(go.Scatter(
                x=prediction_data['Date'],
                y=prediction_data['Sales'],
                mode='lines+markers',
                name='売上',
                line=dict(color='blue'),
                marker=dict(size=5)
            ))
            
            # 異常値のみ強調表示
            anomaly_fig.add_trace(go.Scatter(
                x=anomaly_data['Date'],
                y=anomaly_data['Sales'],
                mode='markers',
                name='異常値',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x'
                )
            ))
            
            anomaly_fig.update_layout(
                title='異常値の時系列表示',
                xaxis_title='日付',
                yaxis_title='売上高',
                height=500
            )
            
            st.plotly_chart(anomaly_fig, use_container_width=True)
        else:
            st.info('選択した期間内に異常値は検出されませんでした。')
    else:
        st.warning("選択した期間に予測データがありません。")

if __name__ == "__main__":
    main()
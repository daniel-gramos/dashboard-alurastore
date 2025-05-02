import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from functools import reduce
import locale
import numpy as np
import io
from matplotlib import gridspec
import seaborn as sns
import folium
from folium.plugins import HeatMap
from collections import defaultdict

# Configuração inicial
st.set_page_config(page_title="Dashboard AluraStore", layout="wide")

# CSS personalizado para o menu lateral e centralizar conteúdo
st.markdown("""
    <style>
        .css-18e3th9 {
            width: 250px;
            margin-left: auto;
            margin-right: auto;
        }
        .block-container {
            max-width: 1920px;
            margin-left: auto;
            margin-right: auto;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .title-container {
            text-align: center;
            padding: 1rem 0;
        }
        .sidebar .sidebar-content {
            margin-top: 20px;
        }
        .metric-box {
            background-color: #f0f2f6;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            color: #000;
        }
        .metric-box p {
            font-weight: 700;
            font-size: 22px;
        }
        .footer {
            text-align: center;
            padding: 2rem 0 1rem;
            font-size: 0.9rem;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Título centralizado
st.markdown('<div class="title-container" style="margin-top: -100px;"><h1>Dashboard AluraStore</h1></div>', unsafe_allow_html=True)

pages = {
    "Visão Geral das Lojas" : "overview",
    "Análise de Faturamento": "faturamento",
    "Vendas por Categoria": "vendas_categoria",
    "Média de Avaliações das Lojas": "avaliacoes",
    "Produtos Mais e Menos Vendidos": "produtos_vendidos",
    "Frete Médio por Loja": "frete_medio",
    "Mapa de Calor de Vendas": "mapa_calor"
}

if "page" not in st.session_state:
    st.session_state.page = "overview"

st.sidebar.title("Menu")
for label, value in pages.items():
    if st.sidebar.button(label, key=value):
        st.session_state.page = value


@st.cache_data
def carregar_dados():
    urls = [
        "https://raw.githubusercontent.com/daniel-gramos/DesafioAluraStore/refs/heads/main/CSVs/loja_1.csv",
        "https://raw.githubusercontent.com/daniel-gramos/DesafioAluraStore/refs/heads/main/CSVs/loja_2.csv",
        "https://raw.githubusercontent.com/daniel-gramos/DesafioAluraStore/refs/heads/main/CSVs/loja_3.csv",
        "https://raw.githubusercontent.com/daniel-gramos/DesafioAluraStore/refs/heads/main/CSVs/loja_4.csv"
    ]
    lojas_dict = {}
    for i, url in enumerate(urls, start=1):
        loja = pd.read_csv(url)
        loja['Data da Compra'] = pd.to_datetime(loja['Data da Compra'], format='%d/%m/%Y')
        loja['Ano'] = loja['Data da Compra'].dt.year
        lojas_dict[f'Loja {i}'] = loja
    return lojas_dict


lojas = carregar_dados()

if st.session_state.page == "overview":
    st.subheader("📊 Visão Geral da AluraStore")

    todas_lojas = pd.concat(lojas.values(), ignore_index=True)

    faturamento_total = todas_lojas['Preço'].sum()
    total_pedidos = todas_lojas.shape[0]
    frete_medio = todas_lojas['Frete'].mean()
    avaliacao_media = todas_lojas['Avaliação da compra'].mean()
    ticket_medio_total = faturamento_total / total_pedidos

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"<div class='metric-box'><h4>💰 Faturamento Total</h4><p>R${faturamento_total:,.2f}</p></div>",
                  unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'><h4>📦 Total de Pedidos</h4><p>{total_pedidos}</p></div>",
                  unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-box'><h4>🚚 Frete Médio</h4><p>R${frete_medio:.2f}</p></div>",
                  unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-box'><h4>⭐ Avaliação Média</h4><p>{avaliacao_media:.2f}</p></div>",
                  unsafe_allow_html=True)
    col5.markdown(f"<div class='metric-box'><h4>🎫 Ticket Médio</h4><p>R${ticket_medio_total:.2f}</p></div>",
                  unsafe_allow_html=True)

    # Top 5 produtos mais e menos vendidos lado a lado
    st.markdown("### 🏆 Top 5 Produtos Mais e Menos Vendidos")
    col1, col2 = st.columns(2)

    top_produtos = todas_lojas['Produto'].value_counts().head(5)
    col1.markdown("**Mais Vendidos**")
    col1.dataframe(top_produtos.reset_index().rename(columns={"index": "Produto", "count": "Quantidade"}))

    menos_vendidos = todas_lojas['Produto'].value_counts(ascending=True).head(5)
    col2.markdown("**Menos Vendidos**")
    col2.dataframe(menos_vendidos.reset_index().rename(columns={"index": "Produto", "count": "Quantidade"}))

    # Faturamento mensal
    todas_lojas['AnoMes'] = todas_lojas['Data da Compra'].dt.to_period('M')
    faturamento_mensal = todas_lojas.groupby('AnoMes')['Preço'].sum().sort_index()

    fig, ax = plt.subplots(figsize=(10, 3))
    faturamento_mensal.plot(kind='line', marker='o', ax=ax)
    ax.set_title("Faturamento Mensal")
    ax.set_ylabel("R$")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'R${x:,.0f}'.replace(",", ".").replace(".", ",", 1)))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Rodapé
    st.markdown("<div class='footer'>© 2025 By Daniel Ramos</div>", unsafe_allow_html=True)

elif st.session_state.page == "faturamento":
    st.subheader("Análise de Faturamento Anual")

    anos_disponiveis = sorted(list(set(reduce(lambda x, y: x + list(y), [df['Ano'] for df in lojas.values()], []))))
    anos_disponiveis = ['Todos'] + anos_disponiveis
    lojas_disponiveis = list(lojas.keys())

    ano_selecionado = st.selectbox("Selecione o Ano", anos_disponiveis, index=0)
    lojas_selecionadas = st.multiselect("Selecione as Lojas", lojas_disponiveis, default=lojas_disponiveis)

    if not lojas_selecionadas:
        st.warning("Nenhuma loja selecionada. Selecione ao menos uma loja para visualizar o gráfico.")
    else:
        dados_filtrados = {}
        for loja_nome in lojas_selecionadas:
            loja_df = lojas[loja_nome]
            loja_df_filtrada = loja_df if ano_selecionado == 'Todos' else loja_df[loja_df['Ano'] == ano_selecionado]
            dados_filtrados[loja_nome] = loja_df_filtrada

        faturamentos = []
        for nome, df in dados_filtrados.items():
            fat = df.groupby('Ano')['Preço'].sum().reset_index()
            fat.columns = ['Ano', nome]
            faturamentos.append(fat)

        faturamento_total = reduce(lambda left, right: pd.merge(left, right, on='Ano', how='outer'), faturamentos)
        loja_cols = [col for col in faturamento_total.columns if 'Loja' in col]
        faturamento_total['Total'] = faturamento_total[loja_cols].sum(axis=1)
        faturamento_total = faturamento_total.sort_values('Ano')

        st.subheader("Tabela de Faturamento Consolidado")
        locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

        def formatar_valor(valor):
            return locale.currency(valor, grouping=True)

        st.dataframe(
            faturamento_total.style.format(
                {col: lambda x: formatar_valor(x) for col in faturamento_total.columns if col != 'Ano'})
        )

        st.subheader("Gráfico de Faturamento Anual por Loja")

        fig = plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])

        bar_width = 0.15
        anos = faturamento_total['Ano'].astype(str)
        x = np.arange(len(anos))
        cores = plt.get_cmap('Set2').colors

        for i, loja in enumerate(loja_cols):
            ax.bar(x + i * bar_width, faturamento_total[loja], width=bar_width, label=loja,
                   color=cores[i % len(cores)])

        bar_total = ax.bar(x + len(loja_cols) * bar_width, faturamento_total['Total'], width=bar_width,
                           label='Total', color='green')

        ax.bar_label(bar_total, labels=[formatar_valor(v) for v in faturamento_total['Total']])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: formatar_valor(x)))

        ax.set_xticks(x + bar_width * len(loja_cols) / 2)
        ax.set_xticklabels(anos)
        ax.set_title("Faturamento Anual por Loja", fontsize=16)
        ax.set_xlabel("Ano")
        ax.set_ylabel("Faturamento (R$)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()

        plt.tight_layout(h_pad=4)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf, use_container_width=True)

        # Gráfico de Faturamento Mensal
        st.subheader("Faturamento Mensal por Loja")
        fig, ax = plt.subplots(figsize=(12, 6))
        for nome_loja in lojas_selecionadas:
            df_loja = lojas[nome_loja]
            if ano_selecionado != 'Todos':
                df_loja = df_loja[df_loja['Ano'] == ano_selecionado]
            df_loja['Ano-Mês'] = df_loja['Data da Compra'].dt.to_period('M').astype(str)
            faturamento_mensal = df_loja.groupby('Ano-Mês')['Preço'].sum().sort_index()
            ax.plot(faturamento_mensal.index, faturamento_mensal.values, marker='o', label=nome_loja)

        ax.set_title("Evolução do Faturamento Mensal")
        ax.set_xlabel("Ano-Mês")
        ax.set_ylabel("Faturamento (R$)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        # Rodapé
        st.markdown("<div class='footer'>© 2025 By Daniel Ramos</div>", unsafe_allow_html=True)

elif st.session_state.page == "vendas_categoria":
    st.subheader("Vendas por Categoria")

    anos = sorted(list(set(reduce(lambda x, y: x + list(y), [df['Ano'].tolist() for df in lojas.values()], []))))
    anos = ['Todos'] + anos
    meses = ['Todos'] + list(range(1, 13))
    lojas_disponiveis = list(lojas.keys())

    col1, col2 = st.columns(2)
    with col1:
        ano_selecionado = st.selectbox("Selecione o Ano", anos, index=0)
    with col2:
        mes_selecionado = st.selectbox("Selecione o Mês", meses, index=0)

    lojas_selecionadas = st.multiselect("Selecione as Lojas", lojas_disponiveis, default=lojas_disponiveis)

    if not lojas_selecionadas:
        st.warning("Nenhuma loja selecionada. Selecione ao menos uma loja para visualizar os gráficos.")
    else:
        lojas_filtradas = []
        for loja_nome in lojas_selecionadas:
            df = lojas[loja_nome].copy()
            if ano_selecionado != 'Todos':
                df = df[df['Ano'] == ano_selecionado]
            if mes_selecionado != 'Todos':
                df = df[df['Data da Compra'].dt.month == mes_selecionado]
            lojas_filtradas.append(df)

        st.subheader("Quantidade de Produtos Vendidos por Categoria")

        fig1 = plt.figure(figsize=(16, 10))
        gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.4)
        axs1 = [fig1.add_subplot(gs1[i]) for i in range(len(lojas_filtradas))]

        for i, (df, ax) in enumerate(zip(lojas_filtradas, axs1), start=1):
            categorias = df['Categoria do Produto'].value_counts()

            def formata_valor(pct, valores):
                total = sum(valores)
                valor = int(round(pct * total / 100.0))
                return f'{valor}'

            ax.pie(
                categorias,
                labels=[cat.title() for cat in categorias.index],
                autopct=lambda pct: formata_valor(pct, categorias),
                startangle=140,
                textprops={'fontsize': 9}
            )
            ax.set_title(f'Loja {i}', fontsize=14)
            ax.axis('equal')
            total_vendas = categorias.sum()
            ax.annotate(f'Total de Vendas: {total_vendas}', xy=(0.5, -0.1), xycoords='axes fraction',
                        ha='center', va='center', fontsize=12, color='black')

        plt.suptitle('Vendas por Categoria - Cada Loja', fontsize=20)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", bbox_inches="tight")
        st.image(buf1, use_container_width=True)

        st.subheader("Valor Total de Vendas por Categoria")

        fig2 = plt.figure(figsize=(16, 12))
        gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.6)
        axs2 = [fig2.add_subplot(gs2[i]) for i in range(len(lojas_filtradas))]
        cores = plt.cm.Set2(np.linspace(0, 1, 8))

        for i, (df, ax) in enumerate(zip(lojas_filtradas, axs2), start=1):
            cat_valores = df.groupby('Categoria do Produto')['Preço'].sum().sort_values(ascending=False)
            bars = ax.bar(cat_valores.index, cat_valores.values, color=cores[:len(cat_valores)])

            for idx, valor in enumerate(cat_valores.values):
                ax.text(idx, valor + (valor * 0.01), f'R${valor:,.0f}'.replace(",", "."), ha='center', va='bottom',
                        fontsize=9)

            ax.set_title(f'Loja {i}', fontsize=14)
            ax.set_xlabel('Categoria do Produto', fontsize=12)
            ax.set_ylabel('Valor Total de Vendas (R$)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('Valor Total de Vendas por Categoria - Lojas Individuais', fontsize=22)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        st.image(buf2, use_container_width=True)

        # Rodapé
        st.markdown("<div class='footer'>© 2025 By Daniel Ramos</div>", unsafe_allow_html=True)

elif st.session_state.page == "avaliacoes":
    st.subheader("Média de Avaliações das Lojas")

    anos_disponiveis = sorted(list(set(reduce(lambda x, y: x + list(y), [df['Ano'].tolist() for df in lojas.values()], []))))
    anos_disponiveis = ['Todos'] + anos_disponiveis
    lojas_disponiveis = list(lojas.keys())

    col1, col2 = st.columns(2)
    with col1:
        ano_selecionado = st.selectbox("Selecione o Ano", anos_disponiveis, index=0)
    with col2:
        lojas_selecionadas = st.multiselect("Selecione as Lojas", lojas_disponiveis, default=lojas_disponiveis)

    if not lojas_selecionadas:
        st.warning("Nenhuma loja selecionada. Selecione ao menos uma loja para visualizar os dados.")
    else:
        medias_avaliacao = []
        for nome_loja in lojas_selecionadas:
            df_loja = lojas[nome_loja]
            if ano_selecionado != 'Todos':
                df_loja = df_loja[df_loja['Ano'] == ano_selecionado]
            media = df_loja['Avaliação da compra'].mean()
            medias_avaliacao.append((nome_loja, round(media, 2)))

        df_avaliacoes = pd.DataFrame(medias_avaliacao, columns=["Loja", "Média de Avaliação"])
        st.dataframe(df_avaliacoes)

        st.subheader("Gráfico de Médias de Avaliação por Loja")
        fig, ax = plt.subplots(figsize=(10, 6))
        lojas_nomes = df_avaliacoes["Loja"]
        medias = df_avaliacoes["Média de Avaliação"]

        bars = ax.bar(lojas_nomes, medias, color=plt.get_cmap('Set2').colors)
        ax.set_ylim(0, 5)
        ax.set_ylabel("Média de Avaliação")
        ax.set_title("Avaliação Média por Loja")
        ax.bar_label(bars, fmt="%.2f", padding=3)

        st.pyplot(fig)

        # Rodapé
        st.markdown("<div class='footer'>© 2025 By Daniel Ramos</div>", unsafe_allow_html=True)

elif st.session_state.page == "produtos_vendidos":
    st.subheader("Produtos Mais e Menos Vendidos")

    anos_disponiveis = sorted(list(set(reduce(lambda x, y: x + list(y), [df['Ano'].tolist() for df in lojas.values()], []))))
    anos_disponiveis = ['Todos'] + anos_disponiveis
    lojas_disponiveis = list(lojas.keys())

    col1, col2 = st.columns(2)
    with col1:
        ano_selecionado = st.selectbox("Selecione o Ano", anos_disponiveis, index=0, key="ano_produtos")
    with col2:
        lojas_selecionadas = st.multiselect("Selecione as Lojas", lojas_disponiveis, default=lojas_disponiveis, key="lojas_produtos")

    if not lojas_selecionadas:
        st.warning("Nenhuma loja selecionada. Selecione ao menos uma loja para visualizar os dados.")
    else:
        produtos_vendas = []
        lojas_filtradas = {nome: df[df['Ano'] == ano_selecionado] if ano_selecionado != 'Todos' else df for nome, df in lojas.items() if nome in lojas_selecionadas}

        for nome_loja, loja in lojas_filtradas.items():
            contagem_produtos = loja['Produto'].value_counts()
            produto_mais_vendido = contagem_produtos.idxmax()
            quantidade_mais_vendido = contagem_produtos.max()
            produto_menos_vendido = contagem_produtos.idxmin()
            quantidade_menos_vendido = contagem_produtos.min()

            produtos_vendas.append({
                'Loja': nome_loja,
                'Produto Mais Vendido': produto_mais_vendido,
                'Quantidade Mais Vendido': quantidade_mais_vendido,
                'Produto Menos Vendido': produto_menos_vendido,
                'Quantidade Menos Vendido': quantidade_menos_vendido
            })

        df_produtos_vendas = pd.DataFrame(produtos_vendas)
        st.dataframe(df_produtos_vendas)

        st.subheader("Top 5 Produtos Mais Vendidos por Loja")
        fig, axes = plt.subplots(1, len(lojas_filtradas), figsize=(6 * len(lojas_filtradas), 5))
        if len(lojas_filtradas) == 1:
            axes = [axes]
        for ax, (nome_loja, loja) in zip(axes, lojas_filtradas.items()):
            top5 = loja['Produto'].value_counts().head(5)
            bars = ax.bar(top5.index, top5.values, color=plt.cm.Set2.colors)
            ax.set_title(f'Top 5 Mais Vendidos - {nome_loja}', fontsize=14)
            ax.set_xticklabels(top5.index, rotation=45, ha='right')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, int(height), ha='center')
        st.pyplot(fig)

        st.subheader("Top 5 Produtos Mais Vendidos - Todas as Lojas")
        todas_filtradas = pd.concat(lojas_filtradas.values())
        top5_geral = todas_filtradas['Produto'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(8, 10))
        bars = ax.bar(top5_geral.index, top5_geral.values, color=plt.cm.Set2.colors)
        ax.set_title('Top 5 Produtos Mais Vendidos - Todas as Lojas', fontsize=16)
        ax.set_xticklabels(top5_geral.index, rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, int(height), ha='center')
        st.pyplot(fig)

        st.subheader("Top 5 Produtos Menos Vendidos por Loja")
        fig, axes = plt.subplots(1, len(lojas_filtradas), figsize=(6 * len(lojas_filtradas), 5))
        if len(lojas_filtradas) == 1:
            axes = [axes]
        for ax, (nome_loja, loja) in zip(axes, lojas_filtradas.items()):
            contagem_produtos = loja['Produto'].value_counts(ascending=True)
            top5_menos = contagem_produtos.head(5)
            produtos_top5 = top5_menos.index.tolist()
            produto_categoria_labels = [f"{produto} ({loja.loc[loja['Produto'] == produto, 'Categoria do Produto'].iloc[0]})" for produto in produtos_top5]
            bars = ax.bar(produto_categoria_labels, top5_menos.values, color=plt.cm.Paired.colors)
            ax.set_title(f'Top 5 Menos Vendidos - {nome_loja}')
            ax.set_xticklabels(produto_categoria_labels, rotation=35, ha='right')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, int(height), ha='center')
        st.pyplot(fig)

        st.subheader("Top 5 Produtos Menos Vendidos - Todas as Lojas")
        top5_menos_geral = todas_filtradas['Produto'].value_counts(ascending=True).head(5)
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(top5_menos_geral.index, top5_menos_geral.values, color=plt.cm.Paired.colors)
        ax.set_title('Top 5 Menos Vendidos - Todas as Lojas')
        ax.set_xticklabels(top5_menos_geral.index, rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, int(height), ha='center')
        st.pyplot(fig)

        # Rodapé
        st.markdown("<div class='footer'>© 2025 By Daniel Ramos</div>", unsafe_allow_html=True)

elif st.session_state.page == "frete_medio":
    st.subheader("Frete Médio por Loja")

    todas_lojas = pd.concat(lojas.values(), ignore_index=True)

    # Agrupamento geral por categoria
    resultado_geral = todas_lojas.groupby('Categoria do Produto').agg(
        Quantidade_Vendas=('Frete', 'count'),
        Frete_Medio=('Frete', 'mean'),
        Preco_Medio_Produto=('Preço', 'mean'),
        Frete_Total=('Frete', 'sum')
    ).reset_index()

    resultado_geral['% Frete sobre Preço'] = (resultado_geral['Frete_Medio'] / resultado_geral['Preco_Medio_Produto']) * 100
    colunas_arredondar = ['Frete_Medio', 'Preco_Medio_Produto', 'Frete_Total', '% Frete sobre Preço']
    resultado_geral[colunas_arredondar] = resultado_geral[colunas_arredondar].round(2)

    st.write("### Visão Geral por Categoria")
    st.dataframe(resultado_geral)

    # Frete médio por categoria e loja
    df_frete_por_categoria = pd.concat([
        df.assign(Loja=nome)
        for nome, df in lojas.items()
    ])

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=df_frete_por_categoria,
        x='Categoria do Produto',
        y='Frete',
        hue='Loja',
        estimator='mean',
        ax=ax
    )
    ax.set_title('Frete Médio por Categoria e Loja')
    ax.set_xlabel('Categoria do Produto')
    ax.set_ylabel('Frete Médio (R$)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Loja')
    st.pyplot(fig)

    # Gráfico de dispersão preço médio vs frete médio por loja
    nomes_lojas = list(lojas.keys())
    dados_lojas = []

    for nome_loja in nomes_lojas:
        loja = lojas[nome_loja]
        preco_medio = loja['Preço'].mean()
        frete_medio = loja['Frete'].mean()

        dados_lojas.append({
            'Loja': nome_loja,
            'Preco_Medio': round(preco_medio, 2),
            'Frete_Medio': round(frete_medio, 2)
        })

    df_resumo = pd.DataFrame(dados_lojas)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_resumo, x='Preco_Medio', y='Frete_Medio', hue='Loja', s=200, ax=ax)

    for i in range(len(df_resumo)):
        preco = df_resumo['Preco_Medio'][i]
        frete = df_resumo['Frete_Medio'][i]
        ax.text(
            x=preco,
            y=frete + 0.2,
            s=f"Preço Médio: R${preco:.2f}\nFrete Médio: R${frete:.2f}",
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.6)
        )

    ax.set_title('Comparativo Geral entre Lojas: Preço Médio vs Frete Médio')
    ax.set_xlabel('Preço Médio (R$)')
    ax.set_ylabel('Frete Médio (R$)')
    ax.grid(True)
    st.pyplot(fig)

    # Rodapé
    st.markdown("<div class='footer'>© 2025 By Daniel Ramos</div>", unsafe_allow_html=True)

elif st.session_state.page == "mapa_calor":
    st.subheader("Mapa de Calor de Vendas")

    def obter_regiao_geografica(lat, lon):
        estados = {
            'AC': 'Norte', 'AL': 'Nordeste', 'AM': 'Norte', 'AP': 'Norte', 'BA': 'Nordeste', 'CE': 'Nordeste',
            'DF': 'Centro-Oeste', 'ES': 'Sudeste', 'GO': 'Centro-Oeste', 'MA': 'Nordeste', 'MT': 'Centro-Oeste',
            'MS': 'Centro-Oeste', 'MG': 'Sudeste', 'PA': 'Norte', 'PB': 'Nordeste', 'PR': 'Sul', 'PE': 'Nordeste',
            'PI': 'Nordeste', 'RJ': 'Sudeste', 'RN': 'Nordeste', 'RO': 'Norte', 'RS': 'Sul', 'RR': 'Norte',
            'SC': 'Sul', 'SE': 'Nordeste', 'SP': 'Sudeste', 'TO': 'Norte'
        }

        coordenadas_estados = {
            'AC': (-8.77, -70.55), 'AL': (-9.71, -35.73), 'AM': (-3.07, -61.66), 'AP': (1.41, -51.77),
            'BA': (-12.96, -38.51), 'CE': (-3.71, -38.54), 'DF': (-15.83, -47.86), 'ES': (-19.19, -40.34),
            'GO': (-16.64, -49.31), 'MA': (-2.55, -44.30), 'MT': (-12.64, -55.42), 'MS': (-20.51, -54.54),
            'MG': (-18.10, -44.38), 'PA': (-5.53, -52.29), 'PB': (-7.06, -35.55), 'PR': (-24.89, -51.55),
            'PE': (-8.28, -35.07), 'PI': (-8.28, -43.68), 'RJ': (-22.84, -43.15), 'RN': (-5.22, -36.52),
            'RO': (-11.22, -62.80), 'RS': (-30.01, -51.22), 'RR': (1.89, -61.22), 'SC': (-27.33, -49.44),
            'SE': (-10.90, -37.07), 'SP': (-23.55, -46.64), 'TO': (-10.25, -48.25)
        }

        menor_distancia = float('inf')
        uf_mais_proxima = None
        for uf, coord_ref in coordenadas_estados.items():
            distancia = ((lat - coord_ref[0])**2 + (lon - coord_ref[1])**2) ** 0.5
            if distancia < menor_distancia:
                menor_distancia = distancia
                uf_mais_proxima = uf

        return estados.get(uf_mais_proxima, "Outras")

    lojas_dfs = lojas

    mapa = folium.Map(location=[-14.2, -51.9], zoom_start=5, control_scale=True)
    heat_data = []
    localizacao_vendas = defaultdict(lambda: defaultdict(int))
    regioes_totais = defaultdict(int)

    for nome_loja, df in lojas_dfs.items():
        for _, row in df.dropna(subset=['lat', 'lon']).iterrows():
            coord = (round(row['lat'], 1), round(row['lon'], 1))
            heat_data.append([row['lat'], row['lon']])
            localizacao_vendas[coord][nome_loja] += 1

    HeatMap(heat_data).add_to(mapa)

    for coord, lojas_info in localizacao_vendas.items():
        nome_regiao = obter_regiao_geografica(coord[0], coord[1])
        total_popup = f"<b>Região: {nome_regiao}</b><br>"
        for loja, qtd in lojas_info.items():
            total_popup += f"{loja}: {qtd} vendas<br>"
            regioes_totais[nome_regiao] += qtd

        folium.Marker(
            location=coord,
            popup=folium.Popup(total_popup, max_width=300),
            icon=folium.Icon(color='red', icon='shopping-cart', prefix='fa'),
        ).add_to(mapa)

    st.components.v1.html(mapa._repr_html_(), height=1100, scrolling=False)

    # Agrupa por lat/lon para cada loja individualmente
    loja1 = lojas['Loja 1']
    loja2 = lojas['Loja 2']
    loja3 = lojas['Loja 3']
    loja4 = lojas['Loja 4']

    vendas_loja1 = loja1.groupby(['lat', 'lon']).size().reset_index(name='Vendas Loja 1')
    vendas_loja2 = loja2.groupby(['lat', 'lon']).size().reset_index(name='Vendas Loja 2')
    vendas_loja3 = loja3.groupby(['lat', 'lon']).size().reset_index(name='Vendas Loja 3')
    vendas_loja4 = loja4.groupby(['lat', 'lon']).size().reset_index(name='Vendas Loja 4')

    # Tabelas lado a lado em HTML
    html = f"""
    <div style="display: flex; justify-content: space-evenly; gap: 20px;">
        <div style="flex: 1;">
            <h4>Loja 1</h4>
            {vendas_loja1.head().to_html(index=False)}
        </div>
        <div style="flex: 1;">
            <h4>Loja 2</h4>
            {vendas_loja2.head().to_html(index=False)}
        </div>
        <div style="flex: 1;">
            <h4>Loja 3</h4>
            {vendas_loja3.head().to_html(index=False)}
        </div>
        <div style="flex: 1;">
            <h4>Loja 4</h4>
            {vendas_loja4.head().to_html(index=False)}
        </div>
    </div>
    """

    st.markdown("### Vendas por Coordenada (Amostra)")
    st.markdown(html, unsafe_allow_html=True)

    # Rodapé
    st.markdown("<div class='footer'>© 2025 By Daniel Ramos</div>", unsafe_allow_html=True)

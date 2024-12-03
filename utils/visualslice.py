from pyecharts.charts import Line, Grid
from pyecharts import options as opts


def generate_trading_graph_with_range(env, output_filename, start_idx, end_idx):
    """
    Generate trading results graphs for a specific data range and save as an HTML file.

    Args:
        env: The trading environment containing historical data.
        output_filename: The name of the output HTML file.
        start_idx: The starting index for slicing the data.
        end_idx: The ending index for slicing the data.
    """
    history = env.envs[0].history

    # 데이터 슬라이싱
    price = history["price"][start_idx:end_idx]
    position = history["position"][start_idx:end_idx]
    cum_reward = history["cum_reward"][start_idx:end_idx]

    # 포지션별로 가격 분리
    long_positions = [price[i] if position[i] == 1 else None for i in range(len(price))]
    short_positions = [price[i] if position[i] == -1 else None for i in range(len(price))]
    neutral_positions = [price[i] if position[i] == 0 else None for i in range(len(price))]

    # 첫 번째 그래프: Price와 Positions
    line1 = (
        Line()
        .add_xaxis(list(range(len(price))))  # X축 범위 변경
        .add_yaxis("Long", long_positions, is_smooth=True, color="red", is_symbol_show=False)
        .add_yaxis("Short", short_positions, is_smooth=True, color="blue", is_symbol_show=False)
        .add_yaxis("Neutral", neutral_positions, is_smooth=True, color="green", is_symbol_show=False)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Trading Results"),
            xaxis_opts=opts.AxisOpts(name="Steps"),
            yaxis_opts=opts.AxisOpts(name="Value"),
            tooltip_opts=opts.TooltipOpts(is_show=False)  # Disable tooltip
        )
    )

    # 두 번째 그래프: Cumulative Reward
    line2 = (
        Line()
        .add_xaxis(list(range(len(price))))  # X축 범위 변경
        .add_yaxis("", cum_reward, is_smooth=True, color="orange", is_symbol_show=False)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="Steps"),
            yaxis_opts=opts.AxisOpts(name="Cumulative Reward"),
            tooltip_opts=opts.TooltipOpts(is_show=False)  # Disable tooltip
        )
    )

    # Grid 레이아웃으로 두 그래프를 위아래로 표시 (크기 조정)
    grid = (
        Grid(init_opts=opts.InitOpts(width="1200px", height="800px"))  # 전체 그래프 크기 설정
        .add(line1, grid_opts=opts.GridOpts(pos_bottom="55%"))  # 위쪽 Price 그래프의 위치
        .add(line2, grid_opts=opts.GridOpts(pos_top="55%"))  # 아래쪽 Reward 그래프의 위치
    )

    # HTML로 저장
    grid.render(+output_filename)
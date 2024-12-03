from gym_trading_env.renderer import Renderer

# 렌더러 초기화 및 실행
renderer = Renderer(render_logs_dir="../render_logs")

# # 시각화에 추가적인 지표 설정 (예시로 SMA 추가)
# renderer.add_line(
#     name="SMA10",
#     function=lambda df: df["close"].rolling(10).mean(),
#     line_options={"width": 1, "color": "purple"}
# )
#
# renderer.add_line(
#     name="SMA20",
#     function=lambda df: df["close"].rolling(20).mean(),
#     line_options={"width": 1, "color": "blue"}
# )

# 렌더링 시작
renderer.run()
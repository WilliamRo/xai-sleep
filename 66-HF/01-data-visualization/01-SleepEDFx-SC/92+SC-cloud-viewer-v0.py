from sc.sc_agent import SCAgent



sca = SCAgent()
sca.report_data_info()
configs = {
  # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
  'show_kde': False,
  # 'show_vector': True,
  # 'scatter_alpha': 0.05,
}
sca.visualize_fp_v1(**configs)

from WebApp_plotter import WebApp_plotter

if __name__ == "__main__":
    # initialize webapp_plotter object
    df_WA = WebApp_plotter()
    
    # plot
    df_WA.make_loc_plot(data_path='./input/summary_SSM.xlsx',
                    seq_csv='./input/loc_sequence.csv',
                    ionization='neg', 
                    y_fixed=True,
                    y_step=8,
                    same_frame=False,
                    chemical_names=None, 
                    save_image=True, 
                    image_title='./output02/ab_vs_loc-ESI_neg-all-dark',
                    dark_mode=False)


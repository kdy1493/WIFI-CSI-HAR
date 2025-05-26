import sys
import os
import autorootcwd
import CSI_To_CSV
import CSV_To_Plot
import Raw_CSI_To_CSV
import Processed_Raw_CSI_Visualizer
import multiprocessing

# # Origin
def run_processing():
    import CSI_To_CSV
    CSI_To_CSV.main()

# Raw
#   import Raw_CSI_To_CSV
#   Raw_CSI_To_CSV.main()


def run_plotting():
    import CSV_To_Plot
    CSV_To_Plot.main()
    # import Processed_Raw_CSI_Visualizer
    # Processed_Raw_CSI_Visualizer.main_visualizer()

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_processing)
    p2 = multiprocessing.Process(target=run_plotting)
    p1.start()
    p2.start()
    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("Terminating...")
        p1.terminate()
        p2.terminate()    
        sys.exit(0)

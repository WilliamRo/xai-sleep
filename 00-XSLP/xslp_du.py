from slp_agent import SLPAgent
import xslp_core as core

def load_data():
    # Load data
    # ...
    # train_set, val_set, test_set = SleepRecord.load(configure=configure)
    train_set, val_set, test_set = SLPAgent.load(configure=None)
    return train_set, val_set, test_set


def configure(data_set):
    return data_set


if __name__ == '__main__':
    train_set, val_set, test_set = load_data()

    # Initiate a pictor
    # p = pictor(title='sleep monitor', figure_size=(15, 9))

    # set plotter
    # m = monitor()
    # p.add_plotter(m)

    # set objects
    # p.objects = sleep_data_list

    # Begin main loop
    # p.show()

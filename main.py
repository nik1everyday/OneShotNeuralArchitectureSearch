from model.train_test import train_test
from model.utils import generate_plot_accuracies, generate_accuracy_table
from model.model import SuperNet


def main():
    print('Training and testing oneshot subnets...')
    one_shot_nas_accuracies = train_test(SuperNet())

    print('Training and testing standalone subnets...')
    standalone_accuracies = []
    for subnet in [(x, y) for x in range(3) for y in range(3)]:
        supernet = SuperNet()
        standalone_accuracies.append(train_test(supernet, subnet=subnet))

    print('Generating oneshot subnets accuracies plot...')
    generate_plot_accuracies('oneshot', one_shot_nas_accuracies,
                             'One Shot Training Progress', 'pics/one-shot-subnets.png')
    print('Generating standalone subnets accuracies plot...')
    generate_plot_accuracies('standalone', standalone_accuracies,
                             'Standalone Training Progress', 'pics/standalone-subnets.png')

    print('Generating oneshot-vs-standalone subnets top-1 accuracies table...')
    generate_accuracy_table(one_shot_nas_accuracies,
                            standalone_accuracies,
                            'pics/oneshot_vs_standalone_table.png')


if __name__ == '__main__':
    main()

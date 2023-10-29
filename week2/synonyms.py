import argparse
import fasttext

directory = r'/workspace/datasets/fasttext/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get synonyms for words from a passed in file')
    general = parser.add_argument_group("general")
    general.add_argument("--input", default=directory+'top_words.txt', help="The input file")
    general.add_argument("--threshold", type=float, default=0.75, help="The threshold for nearest neighbours. Default  0.75")

    args = parser.parse_args()
    input_file = args.input
    threshold = args.threshold

    model = fasttext.load_model(directory + 'title_model.bin')
    with open(input_file) as words_file:
        with open(f'{directory}synonyms.csv', 'w') as output:
            for word in words_file:
                cleaned_word = word.strip()
                nn = model.get_nearest_neighbors(cleaned_word)

                threshold_words = []

                for similarity, sim_word in nn:
                    if similarity > threshold:
                        threshold_words.append(sim_word)

                output_synonyms = "," + ','.join(threshold_words) if len(threshold_words) > 0 else ""

                output.write(f"{cleaned_word}{output_synonyms}\n")

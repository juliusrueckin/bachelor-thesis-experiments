import logging
from telegram import Bot


class TelegramNotifier:

    def __init__(self, config):
        self.token = config["telegram"]["token"]
        self.chat_id = config["telegram"]["chat_id"]
        self.bot = Bot(self.token)
        self.verbose = True if config["telegram"]["verbose"] else False

        self.completed = 0
        self.experiment = config["experiment"]

    def start_experiment(self, run_params):
        """Record the start of the experiment and send telegram message."""
        start_text = "Start {} experiment with {} on {} evaluated through {}!\nHyper-Parameters:\n{}" \
                     .format(self.experiment["experiment_type"], self.experiment["method_name"],
                             self.experiment["dataset_name"], self.experiment["performance_function"],
                             run_params)

        self.send_message(start_text)

    def finished_run(self, results):
        """If verbose, record the completion of one run and send telegram message."""

        if self.verbose:
            complete_text = "Completed run of {} experiment with {} on {} evaluated through {}!\nResults:\n" \
                .format(self.experiment["experiment_type"], self.experiment["method_name"],
                        self.experiment["dataset_name"], self.experiment["performance_function"])
            for measure, result in results.items():
                complete_text += "{}: {}\n".format(measure, result)

            self.send_message(complete_text)

    def finish_experiment(self, run_params):
        """Record the termination of the experiment. Summarize completed and failed runs."""
        finish_text = "Finished {} experiment with {} on {} evaluated through {}!\nHyper-Parameters:\n{}" \
            .format(self.experiment["experiment_type"], self.experiment["method_name"],
                    self.experiment["dataset_name"], self.experiment["performance_function"],
                    run_params)

        self.send_message(finish_text)
            
    def send_message(self, message):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.bot.send_message(chat_id=self.chat_id, text=message)

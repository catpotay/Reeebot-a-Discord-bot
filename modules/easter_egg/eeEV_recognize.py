from ..image_awareness.backend import ClassPredictor
from ..text_generation.backend import GPT2Wrapper


class eeAware():

    recognizer = ClassPredictor()
    
    def recognize(self, url):
        if not ClassPredictor.initialized:
            return ""
        self.recognizer.open_from_url(url)
        pd = self.recognizer.predict()
        # print(url)
        # print(pd)
        if (predicted:=self.recognizer.most_likely(pd, threshold = 0.8))\
            [0] == "I":
            return ""

        if not GPT2Wrapper.initialized:
            return (f'Oh hey you {predicted}!')
        reply = GPT2Wrapper.gen('distilgdex', f'Oh hey you {predicted}! I heard that {predicted}')
        return reply + '\n'
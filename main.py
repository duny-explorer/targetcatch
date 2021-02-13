from flask import Flask, request, render_template
#from model.model import *
from forms import *
from config import Config
from targetcatch import *

app = Flask(__name__)
app.config.from_object(Config)
themes = ['knowledge', 'hard-skills', 'soft-skills', 'tool', 'community', 'subjectivity', 'habits', 'career', 'fixing',
          'art', 'health']
model = TargetCatch()


@app.route('/', methods=["POST", "GET"])
def main():
    form = TextForm()
    if request.method == "GET":
        return render_template('form.html', form=form)
    elif request.method == "POST":
        if form.text.data:
                preds = model.predict(form.text.data)

                data = []

                data.append(preds['label_attainable'])
                del preds['label_attainable']
                data.append(preds['label_time_bound'])
                del preds['label_time_bound']
                data.append(preds['label_education'])
                del preds['label_education']
                data.append(preds['label_unambiguity'])
                del preds['label_unambiguity']

                for i in preds:
                    if preds[i] == "Да":
                        data.append(i.split('_')[-1] if len(i.split('_')) == 3 else "_".join(i.split('_')[-2:]))

                data.append(preds['label_abstraction_level'])

                return render_template('result.html', data=data, l=len(data))

        return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(host='127.0.0.1')


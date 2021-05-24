from flask import request, jsonify, Blueprint, g
import os
from flask_api.model import init_model, run_model
import binascii
from pathlib import Path

models = Blueprint("style_model", __name__)


@models.route('/get_all', methods=["GET"])
def get_all_models():
    res = {"response": []}

    cur_path = Path(os.getcwd())
    # cur_path = cur_path.parent.absolute()
    cur_path = os.path.join(cur_path, "checkpoint")

    # try:
    all_models = os.listdir(cur_path)

    for m in all_models:
        res["response"].append(m)

    if len(all_models) == 0:
        raise Exception("No models available.")
    # except Exception as e:
    #     res = {"error" : e.args}

    return jsonify(res)


@models.route('/apply', methods=["POST"])
def stylize():
    res = {"midi": []}
    content = request.get_json()

    try:
        cur_path = Path(os.getcwd())
        # cur_path = cur_path.parent.absolute()
        cur_path = os.path.join(cur_path, "checkpoint")
        all_models = os.listdir(cur_path)

        if len(all_models) == 0:
            raise Exception("No models available.")

        if not all(i in content for i in ["model", "AtoB", "midi"] ):
            raise Exception("Received json does not contain the required fields: "
                            "'model', 'AtoB', 'midi'.")

        if not (content["model"] in all_models):
            raise Exception(str(content["model"]) +\
                            " is not a valid model name.\nAvailable: " + \
                            ", ".join(all_models) + ".")

        if not (type(content["AtoB"]) == bool):
            raise Exception("'AtoB' must be boolean value, representing the "
                            "direction of translation (A to B, or B to A).")

        if not (type(content["midi"]) == str):
            raise Exception("'midi' midi must be binary string representing midi file.")

        content["midi"] = content["midi"].encode()

        model, args = init_model(content["model"], content["AtoB"])

        styled = run_model(model, content["midi"], args)

        styled = str(styled)[2:-1]
        res["midi"] = styled
    except Exception as e:
        res = {"error": e.args}

    return jsonify(res)


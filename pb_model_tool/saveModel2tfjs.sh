
CUDA_VISIBLE_DEVICES=0 tensorflowjs_converter --input_format=tf_saved_model \
                        --output_format=tfjs_graph_model\
                        --signature_name="custom_signature"\
                        --output_node_names="output" \
                        --saved_model_tags=serve \
                        ../pb_model_Hayao-64 ./web_model_Hayao-64  #input_path #output_path





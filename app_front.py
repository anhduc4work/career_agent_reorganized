import gradio as gr
from app_back import (
    get_or_create_user_thread,
    initialize_config_and_ui,
    update_user_id_dropdown,
    insert_user_thread_to_db,
    edit_message,
    fork_message,
    stream_bot_response,
    handle_user_input,
    remove_checkpoint_from_config,
    refresh_internal_state
)

################## UI ##################
initial_user_id = "Default User"
initial_thread_id = "1"
insert_user_thread_to_db(initial_user_id, initial_thread_id)

with gr.Blocks(fill_width=True) as demo:
    
    gr.Markdown("# \n\n\n ___", height=40)
    gr.Markdown("# Career Agent")
    
    with gr.Tab("Chat"):
        # with gr.Row():
            # with gr.Column(scale=1):
        with gr.Row():
            user_choices_state = gr.State([initial_user_id])
            user_id = gr.Dropdown(
                value=initial_user_id, choices=[initial_user_id],
                label="User ID", interactive=True, allow_custom_value=True,
            )
            add_user = gr.Button("+", scale=0, variant="primary")

        # with gr.Row():
            thread_choices_state = gr.State([initial_thread_id])
            thread_id = gr.Dropdown(
                value=initial_thread_id, choices=[initial_thread_id],
                label="Thread ID", interactive=True
            )
            add_thread = gr.Button("+", scale=0, variant="stop")

        config = gr.JSON(visible=False, value={"configurable": {"thread_id": initial_thread_id, "user_id": initial_user_id}})
        
        gr.Markdown("___"*40, height=40)
        
        with gr.Row():
            
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(type="messages", show_copy_button=True, editable="user")
                msg = gr.MultimodalTextbox(file_types=[".pdf"], show_label=False, placeholder="Input chat")

    with gr.Tab("Underthehood") as tab2:
        with gr.Column():
            cross_thread_info = gr.Textbox(label="User Info (Cross Thread)", interactive=False, visible=True)
            single_thread_summary = gr.Textbox(label="Thread Summary", interactive=False, visible=True)

        with gr.Row():
            cv_text = gr.Textbox(label="CV Content", interactive=True, visible=True)
            new_cv_text = gr.Textbox(label="Reviewed CV", interactive=False, visible=True)

        cp = gr.HighlightedText(
            label="Diff",
            combine_adjacent=True,
            show_legend=True,
            min_width=800
        )

    ############## FUNCTION HOOKS ##############

    demo.load(get_or_create_user_thread, [user_id], [thread_id, thread_choices_state]).\
        then(initialize_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])

    add_user.click(update_user_id_dropdown, [user_choices_state], [user_id, user_choices_state]).\
        then(get_or_create_user_thread, [user_id], [thread_id, thread_choices_state]).\
        then(initialize_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])

    add_thread.click(lambda choices: update_user_id_dropdown(choices), [thread_choices_state], [thread_id, thread_choices_state]).\
        then(lambda u, t: insert_user_thread_to_db(u, t), [user_id, thread_id]).\
        then(initialize_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])

    user_id.input(get_or_create_user_thread, [user_id], [thread_id, thread_choices_state]).\
        then(update_user_id_dropdown, [user_choices_state, user_id], [user_id, user_choices_state]).\
        then(initialize_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])

    thread_id.select(initialize_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])

    chatbot.edit(edit_message, chatbot, chatbot).\
        then(fork_message, [config, chatbot], [config]).\
        then(stream_bot_response, [config, chatbot], [chatbot]).\
        then(remove_checkpoint_from_config, [config], [config])

    msg.submit(handle_user_input, [msg, chatbot], [msg, chatbot]).\
        then(stream_bot_response, [config, chatbot], [chatbot])

    tab2.select(refresh_internal_state, [config], [new_cv_text, single_thread_summary, cross_thread_info])

    # Có thể thêm lại so sánh CV nếu muốn
    # new_cv_text.change(diff_texts, [cv_text, new_cv_text], [cp])

demo.launch(share=True)
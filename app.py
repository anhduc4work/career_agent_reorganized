import gradio as gr
from app_back_func import (
    get_or_create_user_thread,
    initialize_config_and_ui,
    update_user_id_dropdown,
    insert_user_thread_to_db,
    edit_message,
    fork_message,
    stream_bot_response,
    handle_user_input,
    remove_checkpoint_from_config,
    refresh_internal_state,
    show_component,
    enable_button,
    diff_texts
)

from app_testcase_func import (
    demo_review_cv_tool,
    demo_score_jds_tool,
    demo_search_by_query_tool,
    demo_upload_cv_and_search_tool,
    demo_analyze_market_tool
)



################## ------------- UI ------------- ##################
initial_user_id = "Default User"
initial_thread_id = "1"
insert_user_thread_to_db(initial_user_id, initial_thread_id)

with gr.Blocks(fill_width=True) as demo:
    
    gr.Markdown("# \n\n\n ___", height=40)
    gr.Markdown("# Career Agent")
    
    with gr.Tab("Chat"):
        

        config = gr.JSON(visible=False, value={"configurable": {"thread_id": initial_thread_id, "user_id": initial_user_id}})
        
        
        with gr.Row():
            with gr.Column(scale = 1, variant="panel"):
          
                user_choices_state = gr.State([initial_user_id])
                thread_choices_state = gr.State([initial_thread_id])
                
                with gr.Row():
                    user_id = gr.Dropdown(
                        value=initial_user_id, choices=[initial_user_id],
                        label="User ID", interactive=True, allow_custom_value=True,
                    )
                    add_user = gr.Button("+", variant="stop")
                    
                with gr.Row():
                    thread_id = gr.Dropdown(
                        value=initial_thread_id, choices=[initial_thread_id],
                        label="Thread ID", interactive=True
                    )
                    add_thread = gr.Button("+", variant="stop")
                
            with gr.Column(scale=4, variant="compact"):
                chatbot = gr.Chatbot(type="messages", show_copy_button=True, editable="user", height=700, resizable = True)
                with gr.Row():
                    THINK_FLAG = gr.Checkbox(label="No Thinking", scale=0)
                    msg = gr.MultimodalTextbox(file_types=[".pdf"], show_label=False, placeholder="Input chat")
        
        gr.Markdown("___"*40, height=40)
        gr.Markdown("## Suggestion", height=40)
        
        with gr.Row():
            demo_upload_cv_and_search_button = gr.Button("1, Demo upload CV and search", variant="primary")
            demo_search_by_query_button = gr.Button("2, Demo search job query", variant="primary")
            demo_score_jds_button = gr.Button("3, Demo score job descriptions", interactive = False)
            demo_review_cv_button = gr.Button("4, Demo review cv", interactive = False)
            demo_analyze_market_button = gr.Button("5, Demo analyze job market", variant="primary")
        
    with gr.Tab("Underthehood") as tab2:
        with gr.Column():
            cross_thread_info = gr.Textbox(label="User Info (Cross Thread)", interactive=False, visible=True, lines= 5)
            single_thread_summary = gr.Textbox(label="Thread Summary", interactive=False, visible=True, lines= 5)
        
        gr.Markdown("# Recorded Job")
        with gr.Column():
            jds = gr.State([])
            
            @gr.render(inputs=jds)
            def render_todos(jd_list):
                for jd in jd_list:
                                        
                    gr.Markdown(f"### {jd.get('id','')} {jd.get('metadata', {}).get('workingtime','')} {jd.get('metadata', {}).get('position','')} [Job Link]({jd.get('metadata', {}).get('Link','')})",
                                container=True)
                    # gr.Textbox(label="Content", value = jd['page_content'][:200], show_label=False, container=False)
                    
            
        
        with gr.Row():
            cv_text = gr.Textbox(label="CV Content", interactive=False, visible=True, lines = 50, max_lines=50)
            new_cv_text = gr.Markdown(label="Reviewed CV", visible=True, height = 600, show_copy_button=True)

        cp = gr.HighlightedText(
            label="Diff",
            combine_adjacent=True,
            show_legend=True,
            min_width=800,
            color_map={"+": "red", "-": "green"},
        )

    ############## ------------- FUNCTION HOOKS ------------- ##############

    demo.load(get_or_create_user_thread, [user_id], [thread_id, thread_choices_state]).\
        then(initialize_config_and_ui, [thread_id, user_id], [chatbot, config])

    add_user.click(update_user_id_dropdown, [user_choices_state], [user_id, user_choices_state]).\
        then(get_or_create_user_thread, [user_id], [thread_id, thread_choices_state]).\
        then(initialize_config_and_ui, [thread_id, user_id], [chatbot, config])

    add_thread.click(lambda choices: update_user_id_dropdown(choices), [thread_choices_state], [thread_id, thread_choices_state]).\
        then(lambda u, t: insert_user_thread_to_db(u, t), [user_id, thread_id]).\
        then(initialize_config_and_ui, [thread_id, user_id], [chatbot, config]).\
        then(refresh_internal_state, [config], [cv_text, new_cv_text, single_thread_summary, cross_thread_info, jds])

    user_id.input(get_or_create_user_thread, [user_id], [thread_id, thread_choices_state]).\
        then(update_user_id_dropdown, [user_choices_state, user_id], [user_id, user_choices_state]).\
        then(initialize_config_and_ui, [thread_id, user_id], [chatbot, config])

    thread_id.select(initialize_config_and_ui, [thread_id, user_id], [chatbot, config]).\
        then(refresh_internal_state, [config], [cv_text, new_cv_text, single_thread_summary, cross_thread_info, jds])
        

    chatbot.edit(edit_message, chatbot, chatbot).\
        then(fork_message, [config, chatbot], [config]).\
        then(stream_bot_response, [config, chatbot, THINK_FLAG], [chatbot]).\
        then(remove_checkpoint_from_config, [config], [config])

    msg.submit(handle_user_input, [msg, chatbot], [msg, chatbot]).\
        then(stream_bot_response, [config, chatbot, THINK_FLAG], [chatbot]).\
            then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])

    tab2.select(refresh_internal_state, [config], [cv_text, new_cv_text, single_thread_summary, cross_thread_info, jds]).\
        then(diff_texts, [cv_text, new_cv_text], cp)
        

    # Có thể thêm lại so sánh CV nếu muốn
    # new_cv_text.change(diff_texts, [cv_text, new_cv_text], [cp])
    
    ######################## ------------- TEST CASE ------------- #########################
    demo_upload_cv_and_search_button.click(demo_upload_cv_and_search_tool, [chatbot], [chatbot]).\
        then(stream_bot_response, [config, chatbot, THINK_FLAG], [chatbot]).\
            then(enable_button, outputs=[demo_score_jds_button]).\
                then(enable_button, outputs=[demo_review_cv_button]).\
                    then(refresh_internal_state, [config], [cv_text, new_cv_text, single_thread_summary, cross_thread_info, jds]).\
                        then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])
                        
                    
                
    demo_search_by_query_button.click(demo_search_by_query_tool, outputs = [msg]).\
        then(handle_user_input, [msg, chatbot], [msg, chatbot]).\
        then(stream_bot_response, [config, chatbot, THINK_FLAG], [chatbot]).\
        then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])
        
    demo_score_jds_button.click(demo_score_jds_tool, [jds], [msg]).\
        then(handle_user_input, [msg, chatbot], [msg, chatbot]).\
        then(stream_bot_response, [config, chatbot, THINK_FLAG], [chatbot]).\
        then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])
        
    demo_review_cv_button.click(demo_review_cv_tool, [jds], [msg]).\
        then(handle_user_input, [msg, chatbot], [msg, chatbot]).\
        then(stream_bot_response, [config, chatbot, THINK_FLAG], [chatbot]).\
        then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])
    
    demo_analyze_market_button.click(demo_analyze_market_tool, None, [msg]).\
        then(handle_user_input, [msg, chatbot], [msg, chatbot]).\
        then(stream_bot_response, [config, chatbot, THINK_FLAG], [chatbot]).\
        then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])

demo.launch(share=True)
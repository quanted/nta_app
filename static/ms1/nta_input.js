function define_functions(){

    //slider label function
    $(function slider_label()
    {
        $('.slider_bar').on('input change', function(){
            $(this).next($('.slider_label')).html(this.value);
        });
        $('.slider_label').each(function(){
            var value = $(this).prev().attr('value');
            $(this).html(value);
        });
    }).trigger('input change');
    
    $("#id_test_files").change(function() {
        if($(this).val() == 'yes') {                
            $("#id_pos_input").attr("disabled", "disabled");
            $("#id_neg_input").attr("disabled", "disabled");
        }
        else {
            $("#id_pos_input").removeAttr("disabled");
            $("#id_neg_input").removeAttr("disabled");
        }
    });

}


$(document).ready(function(){
    $("#id_min_replicate_hits").before("<p>");
    $("#id_min_replicate_hits").after("<span  class='slider_label'></span></p>");
    $("#id_parent_ion_mass_accuracy").before("<p>");
    $("#id_parent_ion_mass_accuracy").after("<span  class='slider_label'></span></p>");
    define_functions();
    //$("#id_min_replicate_hits").after("<p> text</p>");
});

;(function($) {
    "use strict";
    
    var navbar=$('.main_menu_area');
    /*-------------------------------------------------------------------------------
	  Navbar 
	-------------------------------------------------------------------------------*/

	navbar.affix({
	  offset: {
	    top: 1
	  }
	});


	navbar.on('affix.bs.affix', function() {
		if (!navbar.hasClass('affix')){
			navbar.addClass('animated slideInDown');
		}
	});

	navbar.on('affixed-top.bs.affix', function() {
	  	navbar.removeClass('animated slideInDown');
	  	
	});
    
    
    /*----------------------------------------------------*/
    /*  Home Slider Bg
    /*----------------------------------------------------*/
    
    var slider_text = $('.text_slider');
    function text_slider(){
        if ( slider_text.length ){
            slider_text.owlCarousel({
                loop: false,
                margin: 0,
                dots: true,
                autoplay: true,
                mouseDrag: true,
                touchDrag: true,
                animateOut: 'slideOutUp',
                animateIn: 'fadeInUp',
                navSpeed: 500,
                items: 1,
                smartSpeed: 2500,
            })
        }
    }
    text_slider();
    
    /*----------------------------------------------------*/
    /*  Home Slider Text
    /*----------------------------------------------------*/
    var slider_bg = $('.slider_bg');
    function home_slider(){
        if ( slider_bg.length ){
            slider_bg.owlCarousel({
                loop: false,
                margin: 0,
                dots: true,
                autoplay: true,
                mouseDrag: true,
                touchDrag: true,
                items: 1,
                smartSpeed: 2500,
            })
        }
    }
    home_slider();
    
    /*----------------------------------------------------*/
    /*  Home Slider Next Prev
    /*----------------------------------------------------*/
    $('.home_screen_nav .testi_next').on('click', function () {
        slider_text.trigger('next.owl.carousel');
        slider_bg.trigger('next.owl.carousel');
    });
    $('.home_screen_nav .testi_prev').on('click', function () {
        slider_text.trigger('prev.owl.carousel');
        slider_bg.trigger('prev.owl.carousel');
    });
    
    /*----------------------------------------------------*/
    /*  Home Slider Click
    /*----------------------------------------------------*/
    slider_text.on('translate.owl.carousel', function (property) {
        $('.slider_bg_inner .owl-dots:eq(' + property.page.index + ')').click();
    });
    slider_bg.on('translate.owl.carousel', function (property) {
        $('.text_slider_inner .owl-dots:eq(' + property.page.index + ')').click();
    });
    
    /*----------------------------------------------------*/
    /*  Home Slider Drg
    /*----------------------------------------------------*/
    slider_bg.on('drag.owl.carousel',function(){
        slider_text.trigger('next.owl.carousel');
        slider_bg.trigger('next.owl.carousel');
    });
    slider_text.on('drag.owl.carousel',function(){
        slider_text.trigger('next.owl.carousel');
        slider_bg.trigger('next.owl.carousel');
    });
    
    
    /*======== about slider js =========*/
    
    var about_slider = $(".about-slider");
    function aboutImage(){
        if (about_slider.length){
            about_slider.owlCarousel({
                loop: false,
                margin: 0,
                items: 1,
                dots: true,
                autoplay: true,
                mouseDrag: true,
                touchDrag: true,
                slideSpeed : 1500,
                smartSpeed: 1500,
            });
        }
    }
    aboutImage();
    
    var about_text = $(".text-slider");
    function aboutText(){
        if (about_text.length){
            about_text.owlCarousel({
                loop: false,
                margin: 0,
                items: 1,
                dots: true,
                autoplay: true,
                mouseDrag: true,
                touchDrag: true,
                slideSpeed : 1500,
                smartSpeed: 1500,
            });
        }
    }
    aboutText();
    /*----------------------------------------------------*/
    /*  about Slider Next Prev
    /*----------------------------------------------------*/
    $(".slider_nav .testi_next").on("click", function(){
        about_slider.trigger("next.owl.carousel");
        about_text.trigger("next.owl.carousel");
    });
    $(".slider_nav .testi_prev").on("click", function(){
        about_slider.trigger("prev.owl.carousel");
        about_text.trigger("prev.owl.carousel");
    });
    /*----------------------------------------------------*/
    /*  about Slider Click
    /*----------------------------------------------------*/
    about_text.on('translate.owl.carousel', function (property) {
        $('.about-slider-right .owl-dots:eq(' + property.page.index + ')').click();
    });
    about_slider.on('translate.owl.carousel', function (property) {
        $('.about-slider-left .owl-dots:eq(' + property.page.index + ')').click();
    });
    /*----------------------------------------------------*/
    /*  Home Slider Drg
    /*----------------------------------------------------*/
    about_slider.on('drag.owl.carousel',function(){
        about_text.trigger('next.owl.carousel');
        about_slider.trigger('next.owl.carousel');
    });
    about_text.on('drag.owl.carousel',function(){
        about_text.trigger('next.owl.carousel');
        about_slider.trigger('next.owl.carousel');
    });
    
    /*========== blog slider js ==========*/
    var bl_Slider = $(".bl-slider");
    function blSlider(){
        if (bl_Slider.length){
            bl_Slider.owlCarousel({
                loop: true,
                margin: 0,
                items: 1,
                dots: true,
                autoplay: true,
                mouseDrag: true,
                touchDrag: false,
                nav:true,
                navText: ['<i class="fa fa-angle-left"></i>','<i class="fa fa-angle-right"></i>'],
                smartSpeed: 1500,
            });
        }
    }
    blSlider();
    
    /*========== contact slider js ==========*/
    var contactSlider = $(".contact-slider");
    function contact_Slider(){
        if (contactSlider.length){
            contactSlider.owlCarousel({
                loop: true,
                margin: 0,
                items: 1,
                dots: true,
                animateOut: 'slideOutRight',
                animateIn: 'fadeInLeft',
                autoplay: true,
                mouseDrag: true,
                touchDrag: false,
                nav:false,
                smartSpeed: 2000,
            });
        }
    }
    contact_Slider();
    
    /*========== twitter slider js ==========*/
    var twit_Slider = $(".twitter-slider");
    function twitSlider(){
        if (twit_Slider.length){
            twit_Slider.owlCarousel({
                loop: true,
                margin: 0,
                items: 1,
                dots: true,
                autoplay: true,
                mouseDrag: true,
                touchDrag: false,
                nav:false,
                smartSpeed: 2000,
            });
        }
    }
    twitSlider();
    
    /*----------------------------------------------------*/
    /*  offcanvas menu js
    /*----------------------------------------------------*/
    function offcanvas_menu(){
        if ( $(".nav-button").length ){
            $(".nav-button,.cross").on('click',function(){
                if( $(".offcanvas_menu_click, .nav-button").hasClass('open') ){
                    $(".offcanvas_menu_click, .nav-button, .cross").removeClass('open')
                }
                else{
                    $(".offcanvas_menu_click, .nav-button").addClass('open')
                }
                return false
            });
        }
    }
    offcanvas_menu();
    
    /*======== nav_searchFrom  ========*/
    function searchFrom(){
        if ( $(".search_dropdown").length ){  
             $(".search_dropdown").on("click",function(){
                $(".searchForm").toggleClass('show');
                return false
            });
            $(".form_hide").on("click",function(){
                $(".searchForm").removeClass('show')
            });
        };
    };
    searchFrom();
    /*========End nav_searchFrom  ========*/
    
    /*======== contact form js ========*/
    if ($('.js-ajax-form').length) {
		$('.js-ajax-form').each(function(){
			$(this).validate({
				errorClass: 'error wobble-error',
			    submitHandler: function(form){
		        	$.ajax({
			            type: "POST",
			            url:"mail.php",
			            data: $(form).serialize(),
			            success: function() {
		                	$('.modal').modal('hide');
		                	$('#success').modal('show');
		                },

		                error: function(){
			                $('.modal').modal('hide');
		                	$('#error').modal('show');
			            }
			        });
			    }
			});
		});
	}
    
    /*------------- preloader js --------------*/
     $(window).on("load", function() { // makes sure the whole site is loaded
		$('.loader-container').fadeOut(); // will first fade out the loading animation
		$('.loader').delay(150).fadeOut('slow'); // will fade out the white DIV that covers the website.
		$('body').delay(150).css({'overflow':'visible'})
    });
    
})(jQuery)
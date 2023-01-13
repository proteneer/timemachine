;;; Directory Local Variables            -*- no-byte-compile: t -*-
;;; For more information see (info "(emacs) Directory Variables")

((nil . ((format-all-formatters . (("Python" black)
                                   ("C++" clang-format)
                                   ("Cuda" clang-format)))))
 (python-mode . ((fill-column . 120)
                 (eval . (add-hook 'before-save-hook #'py-isort-before-save)))))

; General identifiers first; specialized captures below take precedence in Zed.
(identifier) @variable
(comment) @comment
(string) @string
(number) @number
(boolean) @boolean
(primitive_type) @type
"Instance" @type

["flow" "rtype" "static" "register" "regfile" "let"] @keyword
["if" "else" "while" "with" "do" "spawn" "return" "sel"] @keyword

(flow_declaration name: (identifier) @function)
(parameter name: (identifier) @variable.parameter)
(call_expression function: (identifier) @function)
(attribute name: (identifier) @attribute)
(directive name: (identifier) @attribute)

((identifier) @variable.special
 (#match? @variable.special "^_(irf|mem|burst_read|burst_write|csr)$"))
(call_expression
 function: (identifier) @function @function.builtin
 (#match? @function.builtin "^(bf16|fp32)_(add|sub|mul|div|sqrt|fma|fmsub|fnmadd|fnmsub|cmp|from_i32|from_u32|to_i32|to_u32|widen|narrow|neg|abs|copysign|min|max|classify)(_flags)?$"))
["$signed" "$unsigned" "$f32" "$f64" "$int" "$uint" "bitcast"] @function @function.builtin

["+" "-" "*" "/" "%" "=" "==" "!=" "<" "<=" ">" ">="
 "&&" "||" "!" "&" "|" "~" "^" "<<" ">>"] @operator
["," ";" ":"] @punctuation.delimiter
["(" ")" "[" "]" "{" "}" "[[" "]]"] @punctuation.bracket
"#" @punctuation.special

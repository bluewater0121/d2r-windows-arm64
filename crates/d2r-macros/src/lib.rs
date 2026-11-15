//! 注册宏：`#[hle("symbol")]`、`#[force("symbol", why = "...")]`、`#[force("symbol", caller)]`。
//!
//! 展开只追加一条 `inventory` 注册项和一层调用日志 wrapper，不改函数本体；符号名与
//! ABI 签名由调用点负责。`caller` 只给需要按客人返回点分支的接管使用（见
//! `d2r_hle::caller`），其它接管不额外读写线程局部槽。

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Error, Ident, ItemFn, LitStr, Token};

struct Args {
    symbol: LitStr,
    why: LitStr,
    /// 在 wrapper 里捕获客人返回点并透传给实现函数。
    caller: bool,
}

impl Parse for Args {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let symbol: LitStr = input.parse()?;
        let mut why = LitStr::new("", symbol.span());
        let mut caller = false;
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            if input.peek(Ident) {
                let key: Ident = input.parse()?;
                match key.to_string().as_str() {
                    "why" => {
                        input.parse::<Token![=]>()?;
                        why = input.parse()?;
                    }
                    "caller" => caller = true,
                    _ => return Err(Error::new(key.span(), "expected `why` or `caller`")),
                }
            } else {
                // 兼容旧写法：`#[force("sym", "为什么")]`。
                why = input.parse()?;
            }
        }
        if !input.is_empty() {
            return Err(Error::new(input.span(), "unexpected trailing tokens"));
        }
        Ok(Self { symbol, why, caller })
    }
}

fn expand(attr: TokenStream, item: TokenStream, force: bool) -> TokenStream {
    let args = match syn::parse::<Args>(attr) {
        Ok(args) => args,
        Err(err) => return err_tokens(err),
    };
    let func: ItemFn = match syn::parse(item.clone()) {
        Ok(func) => func,
        Err(err) => return err_tokens(err),
    };
    let ident = &func.sig.ident;
    let symbol = args.symbol;
    let why = args.why;
    let kind = if force {
        quote!(::d2r_abi::HleKind::Force)
    } else {
        quote!(::d2r_abi::HleKind::Normal)
    };

    // 每个接管函数外面再包一层，只在 debug 级别打印一次符号名：实机排查「游戏走到
    // 哪一步就退出」时，用 `--log debug` 就能看到完整的 HLE 调用序列（默认 info 级别
    // 无额外输出）。转发参数保留原 ABI 与返回类型。
    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        match arg {
            syn::FnArg::Receiver(_) => return err_tokens(Error::new(ident.span(), "不接受 self 参数")),
            syn::FnArg::Typed(typed) => match &*typed.pat {
                syn::Pat::Ident(pat) => params.push(pat.ident.clone()),
                _ => return err_tokens(Error::new(ident.span(), "参数必须是简单标识符")),
            },
        }
    }
    let vis = &func.vis;
    let ret = &func.sig.output;
    let inputs = &func.sig.inputs;
    let wrapper = syn::Ident::new(&format!("__hle_trace_{ident}"), ident.span());
    let counter = syn::Ident::new(&format!("__hle_calls_{ident}"), ident.span());

    // `caller`：进入实现函数前捕获、退出后恢复。两次调用都在 wrapper 内，
    // 因此实现函数读到的永远是「本次客人调用」的返回点。
    let (prologue, epilogue) = if args.caller {
        (
            quote! {
                let __hle_prev_lr = ::d2r_hle::caller::enter(
                    ::d2r_hle::caller::from_frame(::d2r_hle::caller::frame_pointer())
                );
            },
            quote! { ::d2r_hle::caller::leave(__hle_prev_lr); },
        )
    } else {
        (quote! {}, quote! {})
    };

    let generated = quote! {
        #vis extern "C" fn #wrapper(#inputs) #ret {
            #prologue
            // 只在前 8 次与每 4096 次打一条 debug：像 `GetReleasedAudioOutBuffer`
            // 这种每 10ms 被客人成百上千次轮询的函数，逐条记录会把日志刷到几十 MB，
            // 真正有用的错误反而被埋掉。计数是每个符号独立的。
            #[allow(non_upper_case_globals)]
            static #counter: ::core::sync::atomic::AtomicU32 =
                ::core::sync::atomic::AtomicU32::new(0);
            let __hle_calls = #counter.fetch_add(1, ::core::sync::atomic::Ordering::Relaxed) + 1;
            let __hle_result = #ident(#(#params),*);
            #epilogue
            if __hle_calls <= 8 || __hle_calls & 0xFFF == 0 {
                ::d2r_diag::debug!(
                    ::d2r_diag::cat::HLE,
                    "调用 {} 线程={} 次数={} 返回={:#018x}",
                    #symbol,
                    ::d2r_diag::guest_thread(),
                    __hle_calls,
                    ::d2r_abi::HleRetBits::hle_bits(&__hle_result)
                );
            }
            __hle_result
        }
        ::d2r_hle::submit! {
            ::d2r_hle::Registration(::d2r_abi::HleDef {
                symbol: #symbol,
                func: ::d2r_abi::HleFn::from_const_ptr(#wrapper as *const ()),
                why: #why,
                kind: #kind,
            })
        }
    };
    let mut output: proc_macro2::TokenStream = item.into();
    output.extend(generated);
    output.into()
}

fn err_tokens(err: Error) -> TokenStream {
    err.to_compile_error().into()
}

/// HLE 实现。注册为 `HleKind::Normal`。
#[proc_macro_attribute]
pub fn hle(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand(attr, item, false)
}

/// 强制接管实现。注册为 `HleKind::Force`，优先于 HLE 查找。
#[proc_macro_attribute]
pub fn force(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand(attr, item, true)
}

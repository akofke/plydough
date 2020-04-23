use nom::branch::alt;
use nom::bytes::complete::take_while1;
use nom::bytes::streaming::{tag, take_till1};
use nom::character::is_digit;
use nom::character::streaming::{line_ending, not_line_ending, space1};
use nom::combinator::{map, opt, value};
use nom::error::{make_error, ErrorKind};
use nom::multi::separated_list1;
use nom::sequence::{separated_pair, terminated};
use nom::Err::Error;
use nom::{IResult, Parser};

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Header {
    pub format: Format,
    pub version: String,
    pub elements: Vec<ElementDecl>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Format {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PropertyType {
    Char,
    Uchar,
    Short,
    Ushort,
    Int,
    Uint,
    Float,
    Double,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PropertyDecl {
    Single {
        name: String,
        ty: PropertyType,
    },
    List {
        name: String,
        ty: PropertyType,
        length_ty: PropertyType,
    },
}

impl PropertyDecl {
    pub fn new_single(name: String, ty: PropertyType) -> Self {
        PropertyDecl::Single { name, ty }
    }

    pub fn new_list(name: String, ty: PropertyType, length_ty: PropertyType) -> Self {
        PropertyDecl::List {
            name,
            ty,
            length_ty,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            PropertyDecl::Single { name, .. } => name,
            PropertyDecl::List { name, .. } => name,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ElementDecl {
    pub name: String,
    pub count: usize,
    pub properties: Vec<PropertyDecl>,
}

pub fn header(input: &[u8]) -> IResult<&[u8], Header> {
    let (input, _) = terminated(tag(b"ply"), line_ending)(input)?;
    let (input, _) = tag_and_ws(b"format")(input)?;
    let (input, (format, version)) =
        terminated(separated_pair(format, space1, not_line_ending), line_ending)(input)?;
    let comment_or_element = alt((value(None, comment), map(element, |el| Some(el))));
    let (input, elements) = terminated(
        separated_list1(line_ending, comment_or_element),
        line_ending,
    )(input)?;
    let (rest, _) = terminated(tag("end_header"), line_ending)(input)?;
    let elements = elements.into_iter().flatten().collect();
    let header = Header {
        format,
        version: String::from_utf8(version.to_vec()).unwrap(),
        elements,
    };
    Ok((rest, header))
}

fn format(input: &[u8]) -> IResult<&[u8], Format> {
    alt((
        tag(b"ascii").map(|_| Format::Ascii),
        tag(b"binary_little_endian").map(|_| Format::BinaryLittleEndian),
        tag(b"binary_big_endian").map(|_| Format::BinaryBigEndian),
    ))(input)
}

fn tag_and_ws<'a>(token: &'static [u8]) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], &'a [u8]> {
    terminated(tag(token), space1)
}

fn identifier(input: &[u8]) -> IResult<&[u8], &[u8]> {
    take_till1(|b: u8| b.is_ascii_whitespace())(input)
}

fn comment(input: &[u8]) -> IResult<&[u8], ()> {
    let (rest, _) = separated_pair(tag(b"comment"), space1, not_line_ending)(input)?;
    Ok((rest, ()))
}

fn integer(input: &[u8]) -> IResult<&[u8], usize> {
    let (rest, num) = take_while1(is_digit)(input)?;
    match std::str::from_utf8(num).unwrap().parse::<usize>() {
        Ok(num) => Ok((rest, num)),
        Err(_e) => Err(Error(make_error(input, ErrorKind::ParseTo))),
    }
}

fn element_decl(input: &[u8]) -> IResult<&[u8], (&[u8], usize)> {
    let (input, _) = tag_and_ws(b"element")(input)?;
    separated_pair(identifier, space1, integer)(input)
}

fn element(input: &[u8]) -> IResult<&[u8], ElementDecl> {
    let (input, (name, count)) = terminated(element_decl, line_ending)(input)?;
    let (rest, properties) = separated_list1(line_ending, property_decl)(input)?;
    let element = ElementDecl {
        name: String::from_utf8(name.to_vec()).unwrap(),
        count,
        properties,
    };
    Ok((rest, element))
}

fn property_type(input: &[u8]) -> IResult<&[u8], PropertyType> {
    alt((
        value(PropertyType::Char, tag("char")),
        value(PropertyType::Uchar, tag("uchar")),
        value(PropertyType::Short, tag("short")),
        value(PropertyType::Ushort, tag("ushort")),
        value(PropertyType::Int, tag("int")),
        value(PropertyType::Uint, tag("uint")),
        value(PropertyType::Float, tag("float")),
        value(PropertyType::Double, tag("double")),
    ))(input)
}

// TODO: Propagate utf8 errors
fn property_decl(input: &[u8]) -> IResult<&[u8], PropertyDecl> {
    let (input, _) = tag_and_ws(b"property")(input)?;
    let (input, list_decl) = opt(terminated(prop_list_decl, space1))(input)?;
    separated_pair(property_type, space1, identifier)
        .map(|(ty, id)| {
            list_decl.map_or_else(
                || PropertyDecl::Single {
                    name: String::from_utf8(id.to_owned()).unwrap(),
                    ty,
                },
                |length_ty| PropertyDecl::List {
                    name: String::from_utf8(id.to_owned()).unwrap(),
                    ty,
                    length_ty,
                },
            )
        })
        .parse(input)
}

fn prop_list_decl(input: &[u8]) -> IResult<&[u8], PropertyType> {
    separated_pair(tag(b"list"), space1, property_type)
        .map(|(_, ty)| ty)
        .parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header() {
        let file = include_bytes!("test_files/simple_ascii.ply");
        let expected = Header {
            format: Format::Ascii,
            version: "1.0".to_string(),
            elements: vec![
                ElementDecl {
                    name: "vertex".to_string(),
                    count: 8,
                    properties: vec![
                        PropertyDecl::new_single("x".to_string(), PropertyType::Float),
                        PropertyDecl::new_single("y".to_string(), PropertyType::Float),
                        PropertyDecl::new_single("z".to_string(), PropertyType::Float),
                    ],
                },
                ElementDecl {
                    name: "face".to_string(),
                    count: 6,
                    properties: vec![PropertyDecl::new_list(
                        "vertex_index".to_string(),
                        PropertyType::Int,
                        PropertyType::Uchar,
                    )],
                },
            ],
        };

        let (_rest, header) = header(file)
            .map_err(|e| e.map_input(|input| std::str::from_utf8(input).unwrap()))
            .unwrap();
        assert_eq!(header, expected);
    }
}

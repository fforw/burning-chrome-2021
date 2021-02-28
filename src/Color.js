const colorRegExp = /^(#)?([0-9a-f]+)$/i;

function hex(n)
{
    const s = n.toString(16);

    return s.length === 1 ? "0" + s : s;
}

function hue2rgb(p, q, t){
    if(t < 0) t += 1;
    if(t > 1) t -= 1;
    if(t < 1/6) return p + (q - p) * 6 * t;
    if(t < 1/2) return q;
    if(t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
}

export default class Color
{
    r;
    g;
    b;

    constructor(r,g,b)
    {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    mix(other, ratio, out)
    {
        if (!out)
        {
            out = new Color();
        }
        out.r = (this.r + (other.r - this.r) * ratio)|0;
        out.g = (this.g + (other.g - this.g) * ratio)|0;
        out.b = (this.b + (other.b - this.b) * ratio)|0;

        return out;
    }

    multiply(n, out)
    {
        if (!out)
        {
            out = new Color();
        }

        out.r = this.r * n;
        out.g = this.g * n;
        out.b = this.b * n;
        return out;
    }

    toRGBHex()
    {
        return "#" + hex(this.r) + hex(this.g) + hex(this.b );
    }

    toHex()
    {
        return (this.r << 16) + (this.g << 8) + this.b;
    }

    static validate(color)
    {

        let m;
        if (typeof color !== "string" || !(m = colorRegExp.exec(color)))
        {
            return null;
        }
        const col = m[2];

        if (col.length === 3)
        {
            return new Color(
                parseInt(col[0], 16) * 17,
                parseInt(col[1], 16) * 17,
                parseInt(col[2], 16) * 17
            )
        }
        else if (col.length === 6)
        {
            return new Color(
                parseInt(col.substring(0, 2), 16),
                parseInt(col.substring(2, 4), 16),
                parseInt(col.substring(4, 6), 16)
            )
        }
        else
        {
            return null;
        }
    }

    static from(color, factor = 1.0)
    {
        if (Array.isArray(color))
        {
            const length = color.length;
            const array = new Float32Array(length * 3);

            const f = factor/255;

            let off = 0;
            for (let i = 0; i < length; i++)
            {
                const col = Color.from(color[i]);
                array[off++] = col.r * f;
                array[off++] = col.g * f;
                array[off++] = col.b * f;
            }

            return array;
        }

        const col = Color.validate(color);

        if (!col)
        {
            throw new Error("Invalid color " + color);
        }

        col.r *= factor;
        col.g *= factor;
        col.b *= factor;

        return col;
    }

    static fromHSL(h,s,l)
    {
        let r, g, b;

        if(s <= 0){
            r = g = b = l; // achromatic
        }else{

            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }

        return new Color(
            Math.round(r * 255),
            Math.round(g * 255),
            Math.round(b * 255)
        );
    }
}

